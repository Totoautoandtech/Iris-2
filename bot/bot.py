import os
import asyncio
import logging
import aiohttp
import discord
import yt_dlp
import time
from collections import defaultdict, deque
from discord.ext import commands, tasks
from datetime import datetime

# =========================================================
# LOGS
# =========================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("IRIS-V2")

# =========================================================
# CONFIG — CLÉS API (toutes gratuites)
# =========================================================
TOKEN           = os.getenv("DISCORD_TOKEN_V2")
NEWS_KEY        = os.getenv("NEWS_API_KEY")           # newsapi.org — gratuit
UNSPLASH_KEY    = os.getenv("UNSPLASH_API_KEY")       # unsplash.com — gratuit

# --- IA GRATUITES (par ordre de priorité) ---
GROQ_KEY        = os.getenv("GROQ_API_KEY")           # console.groq.com — 100% gratuit
OPENROUTER_KEY  = os.getenv("OPENROUTER_API_KEY")     # openrouter.ai — modèles gratuits
MISTRAL_KEY     = os.getenv("MISTRAL_API_KEY")        # console.mistral.ai — tier gratuit
HF_KEY          = os.getenv("HF_API_KEY")             # huggingface.co — 100% gratuit
GEMINI_KEY      = os.getenv("GEMINI_API_KEY")         # aistudio.google.com — gratuit

# =========================================================
# ANTI-CRASH — RATE LIMITING 10 000 USERS
# =========================================================
# Max 5 requêtes par utilisateur par minute
RATE_LIMIT_MAX     = 5
RATE_LIMIT_WINDOW  = 60  # secondes
user_requests      = defaultdict(deque)   # {user_id: deque of timestamps}
ai_semaphore       = asyncio.Semaphore(20) # max 20 requêtes IA en parallèle
command_queue      = asyncio.Queue(maxsize=500) # file d'attente globale

def is_rate_limited(user_id: int) -> bool:
    now = time.time()
    timestamps = user_requests[user_id]
    # Nettoyer les vieux timestamps
    while timestamps and timestamps[0] < now - RATE_LIMIT_WINDOW:
        timestamps.popleft()
    if len(timestamps) >= RATE_LIMIT_MAX:
        return True
    timestamps.append(now)
    return False

def get_cooldown_remaining(user_id: int) -> int:
    now = time.time()
    timestamps = user_requests[user_id]
    if not timestamps:
        return 0
    oldest = timestamps[0]
    return max(0, int(RATE_LIMIT_WINDOW - (now - oldest)))

# Cache des réponses IA pour éviter les doublons
response_cache = {}  # {hash(prompt): (response, timestamp)}
CACHE_DURATION = 300  # 5 minutes

def get_cached_response(prompt: str):
    key = hash(prompt)
    if key in response_cache:
        resp, ts = response_cache[key]
        if time.time() - ts < CACHE_DURATION:
            return resp
    return None

def cache_response(prompt: str, response: str):
    response_cache[hash(prompt)] = (response, time.time())
    # Nettoyer le cache si trop grand
    if len(response_cache) > 1000:
        oldest_keys = sorted(response_cache, key=lambda k: response_cache[k][1])[:200]
        for k in oldest_keys:
            del response_cache[k]

# =========================================================
# MOTEUR IA MULTI-PROVIDER (fallback automatique)
# =========================================================
async def ask_ai(prompt: str, system: str = "Tu es Iris, une IA assistante sympathique.") -> str:
    # Vérifier le cache
    cache_key = f"{system[:50]}:{prompt}"
    cached = get_cached_response(cache_key)
    if cached:
        return cached

    async with ai_semaphore:
        # 1. Groq (le plus rapide, gratuit)
        if GROQ_KEY:
            result = await _ask_groq(prompt, system)
            if result:
                cache_response(cache_key, result)
                return result

        # 2. OpenRouter (modèles gratuits)
        if OPENROUTER_KEY:
            result = await _ask_openrouter(prompt, system)
            if result:
                cache_response(cache_key, result)
                return result

        # 3. Mistral (tier gratuit)
        if MISTRAL_KEY:
            result = await _ask_mistral(prompt, system)
            if result:
                cache_response(cache_key, result)
                return result

        # 4. Hugging Face (100% gratuit)
        if HF_KEY:
            result = await _ask_huggingface(prompt, system)
            if result:
                cache_response(cache_key, result)
                return result

        # 5. Gemini (fallback)
        if GEMINI_KEY:
            result = await _ask_gemini(prompt, system)
            if result:
                cache_response(cache_key, result)
                return result

    return "⚠️ Aucune IA disponible. Vérifie tes clés API."

async def _ask_groq(prompt: str, system: str) -> str:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"},
                json={"model": "llama3-8b-8192", "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt}
                ], "max_tokens": 1000}
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data["choices"][0]["message"]["content"]
                elif resp.status == 429:
                    logger.warning("Groq rate limit atteint, passage au prochain provider")
    except Exception as e:
        logger.error(f"Groq error: {e}")
    return None

async def _ask_openrouter(prompt: str, system: str) -> str:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENROUTER_KEY}", "Content-Type": "application/json"},
                json={"model": "meta-llama/llama-3.1-8b-instruct:free", "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt}
                ]}
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"OpenRouter error: {e}")
    return None

async def _ask_mistral(prompt: str, system: str) -> str:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers={"Authorization": f"Bearer {MISTRAL_KEY}", "Content-Type": "application/json"},
                json={"model": "mistral-small-latest", "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt}
                ]}
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"Mistral error: {e}")
    return None

async def _ask_huggingface(prompt: str, system: str) -> str:
    try:
        full_prompt = f"{system}\n\nUser: {prompt}\nAssistant:"
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2",
                headers={"Authorization": f"Bearer {HF_KEY}"},
                json={"inputs": full_prompt, "parameters": {"max_new_tokens": 500}}
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if isinstance(data, list) and data:
                        text = data[0].get("generated_text", "")
                        return text.replace(full_prompt, "").strip()
    except Exception as e:
        logger.error(f"HuggingFace error: {e}")
    return None

async def _ask_gemini(prompt: str, system: str) -> str:
    try:
        from google import genai
        client = genai.Client(api_key=GEMINI_KEY)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"{system}\n\n{prompt}"
        )
        return response.text
    except Exception as e:
        logger.error(f"Gemini error: {e}")
    return None

# =========================================================
# ETAT
# =========================================================
conversation_memory = {}
personalities        = {}
music_queues         = {}
reminders            = []
news_alerts          = {}
custom_commands      = {}
word_filters         = {}
playlists            = {}
current_tracks       = {}  # {guild_id: track}

PERSONALITIES = {
    "normal":  "Tu es Iris, une IA assistante sympathique et utile. Réponds en français.",
    "drole":   "Tu es Iris, une IA très drôle qui fait des blagues et jeux de mots. Réponds en français.",
    "serieux": "Tu es Iris, une IA sérieuse et professionnelle. Réponds de manière concise en français.",
    "kawaii":  "Tu es Iris une IA kawaii qui parle avec des emojis mignons 🌸✨ très enthousiaste ! En français.",
}

intents = discord.Intents.all()
bot = commands.Bot(command_prefix="!v2 ", intents=intents)

# =========================================================
# UTILITAIRES
# =========================================================
async def safe_send(dest, text):
    if not text: return
    for i in range(0, len(text), 1900):
        await dest.send(text[i:i+1900])

def get_queue(guild_id):
    if guild_id not in music_queues:
        music_queues[guild_id] = []
    return music_queues[guild_id]

def search_youtube(query):
    opts = {"format": "bestaudio/best", "noplaylist": True, "quiet": True, "default_search": "ytsearch1"}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(query, download=False)
        if "entries" in info:
            info = info["entries"][0]
        return {
            "url": info["url"], "title": info.get("title", "Inconnu"),
            "thumbnail": info.get("thumbnail"), "duration": info.get("duration", 0),
            "webpage_url": info.get("webpage_url", "")
        }

async def play_next(ctx, vc):
    queue = get_queue(ctx.guild.id)
    if not queue:
        current_tracks.pop(ctx.guild.id, None)
        await ctx.send("✅ File vide, à bientôt !")
        await vc.disconnect()
        return
    track = queue.pop(0)
    current_tracks[ctx.guild.id] = track
    ffmpeg_opts = {"before_options": "-reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5", "options": "-vn"}
    source = discord.FFmpegPCMAudio(track["url"], **ffmpeg_opts)
    vc.play(source, after=lambda e: asyncio.run_coroutine_threadsafe(play_next(ctx, vc), bot.loop))
    m, s = track["duration"] // 60, track["duration"] % 60
    embed = discord.Embed(title="🎵 En cours", description=f"**[{track['title']}]({track['webpage_url']})**", color=discord.Color.green())
    if track["thumbnail"]: embed.set_thumbnail(url=track["thumbnail"])
    embed.add_field(name="⏱ Durée", value=f"{m}:{s:02d}")
    await ctx.send(embed=embed)

# =========================================================
# EVENTS
# =========================================================
@bot.event
async def on_ready():
    logger.info(f"🚀 IRIS V2 connecté : {bot.user}")
    logger.info(f"📡 Présent sur {len(bot.guilds)} serveurs")
    await bot.change_presence(activity=discord.Activity(type=discord.ActivityType.playing, name="Iris V2 🐾 | !v2 aide"))
    check_reminders.start()
    check_news_alerts.start()

@bot.event
async def on_message(message):
    if message.author.bot:
        return
    guild_id = message.guild.id if message.guild else None

    # Filtre de mots
    if guild_id and guild_id in word_filters:
        for mot in word_filters[guild_id]:
            if mot.lower() in message.content.lower():
                await message.delete()
                await message.channel.send(f"⚠️ {message.author.mention} Message supprimé.")
                return

    # Commandes custom
    if guild_id and guild_id in custom_commands:
        if message.content.strip() in custom_commands[guild_id]:
            await message.channel.send(custom_commands[guild_id][message.content.strip()])
            return

    # Mention IA avec rate limiting
    if bot.user and bot.user in message.mentions:
        if is_rate_limited(message.author.id):
            remaining = get_cooldown_remaining(message.author.id)
            await message.channel.send(f"⏳ {message.author.mention} Trop de requêtes ! Attends **{remaining}s** avant de réessayer.")
            return
        prompt = message.content.replace(f"<@{bot.user.id}>", "").replace(f"<@!{bot.user.id}>", "").strip()
        if prompt:
            personality = PERSONALITIES.get(personalities.get(guild_id, "normal"), PERSONALITIES["normal"])
            uid = message.author.id
            if uid not in conversation_memory:
                conversation_memory[uid] = []
            conversation_memory[uid].append(f"User: {prompt}")
            history = "\n".join(conversation_memory[uid][-6:])
            full_prompt = f"Historique:\n{history}\n\nRéponds à la dernière question de l'utilisateur."
            typing_msg = await message.channel.send("🔮 Iris réfléchit...")
            response = await ask_ai(full_prompt, personality)
            conversation_memory[uid].append(f"Iris: {response}")
            await typing_msg.delete()
            await safe_send(message.channel, response)

    await bot.process_commands(message)

# Decorator anti-spam pour toutes les commandes
def anti_spam():
    async def predicate(ctx):
        if is_rate_limited(ctx.author.id):
            remaining = get_cooldown_remaining(ctx.author.id)
            await ctx.send(f"⏳ {ctx.author.mention} Attends **{remaining}s** avant de réessayer.")
            return False
        return True
    return commands.check(predicate)

# =========================================================
# 🤖 IA & CHAT
# =========================================================
@bot.command(name="ask")
@anti_spam()
async def ask(ctx, *, question):
    """!v2 ask <question>"""
    uid = ctx.author.id
    guild_id = ctx.guild.id if ctx.guild else None
    personality = PERSONALITIES.get(personalities.get(guild_id, "normal"), PERSONALITIES["normal"])
    if uid not in conversation_memory:
        conversation_memory[uid] = []
    conversation_memory[uid].append(f"User: {question}")
    history = "\n".join(conversation_memory[uid][-6:])
    msg = await ctx.send("🔮 Iris réfléchit...")
    response = await ask_ai(f"Historique:\n{history}\n\nRéponds.", personality)
    conversation_memory[uid].append(f"Iris: {response}")
    await msg.delete()
    await safe_send(ctx, response)

@bot.command(name="reset")
async def reset_memory(ctx):
    """!v2 reset — efface la mémoire"""
    conversation_memory.pop(ctx.author.id, None)
    await ctx.send("🧹 Mémoire effacée !")

@bot.command(name="resume")
@anti_spam()
async def summarize(ctx):
    """!v2 resume — résumé de conversation"""
    uid = ctx.author.id
    if uid not in conversation_memory or not conversation_memory[uid]:
        return await ctx.send("❌ Aucune conversation en mémoire.")
    history = "\n".join(conversation_memory[uid][-10:])
    response = await ask_ai(f"Résume cette conversation en 3 phrases max :\n{history}")
    await safe_send(ctx, f"📝 **Résumé :**\n{response}")

@bot.command(name="perso")
async def set_personality(ctx, mode: str = "normal"):
    """!v2 perso <normal/drole/serieux/kawaii>"""
    if mode not in PERSONALITIES:
        return await ctx.send(f"❌ Modes : {', '.join(PERSONALITIES.keys())}")
    personalities[ctx.guild.id] = mode
    await ctx.send(f"✅ Personnalité : **{mode}** !")

@bot.command(name="traduire")
@anti_spam()
async def translate(ctx, langue: str, *, texte: str):
    """!v2 traduire <langue> <texte>"""
    response = await ask_ai(f"Traduis uniquement ce texte en {langue}, rien d'autre : {texte}")
    embed = discord.Embed(title=f"🌍 Traduction en {langue}", description=response, color=discord.Color.blue())
    await ctx.send(embed=embed)

# =========================================================
# 🎵 MUSIQUE
# =========================================================
@bot.command(name="play")
@anti_spam()
async def play(ctx, *, search: str):
    """!v2 play <chanson>"""
    if not ctx.author.voice:
        return await ctx.send("❌ Rejoins un salon vocal !")
    vc = ctx.voice_client or await ctx.author.voice.channel.connect()
    if vc.channel != ctx.author.voice.channel:
        await vc.move_to(ctx.author.voice.channel)
    await ctx.send(f"🔍 Recherche **{search}**...")
    try:
        track = await asyncio.get_event_loop().run_in_executor(None, search_youtube, search)
    except Exception as e:
        return await ctx.send(f"⚠️ Erreur : {e}")
    queue = get_queue(ctx.guild.id)
    if vc.is_playing() or vc.is_paused():
        queue.append(track)
        return await ctx.send(f"📋 Ajouté : **{track['title']}** (#{len(queue)})")
    queue.insert(0, track)
    await play_next(ctx, vc)

@bot.command(name="np")
async def now_playing(ctx):
    """!v2 np — chanson en cours"""
    track = current_tracks.get(ctx.guild.id)
    if not track:
        return await ctx.send("❌ Aucune musique en cours.")
    embed = discord.Embed(title="🎵 En cours", description=f"**{track['title']}**", color=discord.Color.green())
    if track["thumbnail"]: embed.set_thumbnail(url=track["thumbnail"])
    await ctx.send(embed=embed)

@bot.command(name="skip")
async def skip(ctx):
    """!v2 skip"""
    if ctx.voice_client and ctx.voice_client.is_playing():
        ctx.voice_client.stop()
        await ctx.send("⏭️ Suivante !")
    else:
        await ctx.send("❌ Aucune musique.")

@bot.command(name="pause")
async def pause(ctx):
    """!v2 pause"""
    if ctx.voice_client and ctx.voice_client.is_playing():
        ctx.voice_client.pause()
        await ctx.send("⏸️ En pause.")
    else:
        await ctx.send("❌ Aucune musique.")

@bot.command(name="reprendre")
async def reprendre(ctx):
    """!v2 reprendre"""
    if ctx.voice_client and ctx.voice_client.is_paused():
        ctx.voice_client.resume()
        await ctx.send("▶️ Reprise !")
    else:
        await ctx.send("❌ Aucune musique en pause.")

@bot.command(name="stop")
async def stop(ctx):
    """!v2 stop"""
    if ctx.voice_client:
        get_queue(ctx.guild.id).clear()
        current_tracks.pop(ctx.guild.id, None)
        await ctx.voice_client.disconnect()
        await ctx.send("⏹️ Arrêté.")
    else:
        await ctx.send("❌ Pas dans un salon vocal.")

@bot.command(name="file")
async def file(ctx):
    """!v2 file — file d'attente"""
    queue = get_queue(ctx.guild.id)
    if not queue:
        return await ctx.send("📋 File vide.")
    embed = discord.Embed(title="📋 File d'attente", color=discord.Color.blue())
    for i, t in enumerate(queue[:10], 1):
        m, s = t["duration"] // 60, t["duration"] % 60
        embed.add_field(name=f"{i}. {t['title']}", value=f"⏱ {m}:{s:02d}", inline=False)
    await ctx.send(embed=embed)

@bot.command(name="volume")
async def volume(ctx, vol: int):
    """!v2 volume <0-200>"""
    if not ctx.voice_client or not ctx.voice_client.source:
        return await ctx.send("❌ Aucune musique.")
    if not 0 <= vol <= 200:
        return await ctx.send("❌ Volume entre 0 et 200.")
    ctx.voice_client.source = discord.PCMVolumeTransformer(ctx.voice_client.source, volume=vol/100)
    await ctx.send(f"🔊 Volume : **{vol}%**")

@bot.command(name="paroles")
@anti_spam()
async def lyrics(ctx, *, chanson: str):
    """!v2 paroles artiste titre"""
    await ctx.send(f"🎤 Recherche paroles : **{chanson}**...")
    parts = chanson.split(" ", 1)
    artist, title = (parts[0], parts[1]) if len(parts) > 1 else (chanson, chanson)
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"https://api.lyrics.ovh/v1/{artist}/{title}") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    text = data.get("lyrics", "")[:1800]
                    embed = discord.Embed(title=f"🎤 {chanson}", description=text, color=discord.Color.purple())
                    return await ctx.send(embed=embed)
    except:
        pass
    await ctx.send("❌ Paroles introuvables. Format : `!v2 paroles artiste titre`")

@bot.command(name="sauver")
async def save_playlist(ctx, nom: str):
    """!v2 sauver <nom> — sauvegarde la file"""
    uid = ctx.author.id
    queue = get_queue(ctx.guild.id)
    if not queue:
        return await ctx.send("❌ File vide.")
    if uid not in playlists:
        playlists[uid] = {}
    playlists[uid][nom] = queue.copy()
    await ctx.send(f"✅ Playlist **{nom}** sauvegardée ({len(queue)} chansons) !")

@bot.command(name="playlist")
async def load_playlist(ctx, nom: str):
    """!v2 playlist <nom> — charge une playlist"""
    uid = ctx.author.id
    if uid not in playlists or nom not in playlists[uid]:
        return await ctx.send("❌ Playlist introuvable.")
    if not ctx.author.voice:
        return await ctx.send("❌ Rejoins un salon vocal !")
    vc = ctx.voice_client or await ctx.author.voice.channel.connect()
    queue = get_queue(ctx.guild.id)
    queue.extend(playlists[uid][nom])
    await ctx.send(f"✅ Playlist **{nom}** chargée ({len(playlists[uid][nom])} chansons) !")
    if not vc.is_playing():
        await play_next(ctx, vc)

@bot.command(name="recommande")
@anti_spam()
async def recommend(ctx, *, genre: str):
    """!v2 recommande <genre>"""
    response = await ask_ai(f"Recommande 5 chansons du genre {genre}. Format : Artiste - Titre. Juste la liste.")
    embed = discord.Embed(title=f"🎵 Recommandations : {genre}", description=response, color=discord.Color.green())
    await ctx.send(embed=embed)

# =========================================================
# 📰 ACTUALITES
# =========================================================
@bot.command(name="news")
@anti_spam()
async def news(ctx, *, sujet: str = "france"):
    """!v2 news <sujet>"""
    if not NEWS_KEY:
        return await ctx.send("❌ NEWS_API_KEY manquante.")
    await ctx.send(f"📰 Recherche : **{sujet}**...")
    url = f"https://newsapi.org/v2/everything?q={sujet}&language=fr&pageSize=5&sortBy=publishedAt&apiKey={NEWS_KEY}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                data = await resp.json()
        articles = [a for a in data.get("articles", []) if a.get("title") and a["title"] != "[Removed]"]
        if not articles:
            return await ctx.send(f"❌ Aucune actualité pour **{sujet}**.")
        for article in articles[:5]:
            embed = discord.Embed(title=article.get("title"), description=article.get("description", ""), url=article.get("url", ""), color=discord.Color.blue())
            if article.get("urlToImage"):
                embed.set_image(url=article["urlToImage"])
            embed.set_footer(text=f"📡 {article.get('source', {}).get('name', '')} — {article.get('publishedAt', '')[:10]}")
            await ctx.send(embed=embed)
            await asyncio.sleep(0.5)
    except Exception as e:
        await ctx.send(f"⚠️ {e}")

@bot.command(name="newsresume")
@anti_spam()
async def news_summary(ctx, *, sujet: str = "france"):
    """!v2 newsresume <sujet> — résumé IA des news"""
    if not NEWS_KEY:
        return await ctx.send("❌ NEWS_API_KEY manquante.")
    url = f"https://newsapi.org/v2/everything?q={sujet}&language=fr&pageSize=5&sortBy=publishedAt&apiKey={NEWS_KEY}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                data = await resp.json()
        articles = [a for a in data.get("articles", []) if a.get("title") and a["title"] != "[Removed]"]
        if not articles:
            return await ctx.send("❌ Aucune actualité.")
        titres = "\n".join([f"- {a['title']}" for a in articles[:5]])
        response = await ask_ai(f"Résume ces actualités sur {sujet} en français, de manière claire et concise :\n{titres}")
        embed = discord.Embed(title=f"📰 Résumé IA : {sujet}", description=response, color=discord.Color.blue())
        await ctx.send(embed=embed)
    except Exception as e:
        await ctx.send(f"⚠️ {e}")

@bot.command(name="alerte")
async def set_alert(ctx, sujet: str, intervalle: int = 60):
    """!v2 alerte <sujet> <minutes>"""
    news_alerts[ctx.guild.id] = {"channel": ctx.channel.id, "sujet": sujet, "interval": intervalle, "last": datetime.now()}
    await ctx.send(f"✅ Alerte **{sujet}** toutes les **{intervalle} min** !")

@bot.command(name="alerteoff")
async def stop_alert(ctx):
    """!v2 alerteoff"""
    news_alerts.pop(ctx.guild.id, None)
    await ctx.send("✅ Alertes désactivées.")

@tasks.loop(minutes=1)
async def check_news_alerts():
    now = datetime.now()
    for guild_id, alert in list(news_alerts.items()):
        diff = (now - alert["last"]).total_seconds() / 60
        if diff >= alert["interval"]:
            news_alerts[guild_id]["last"] = now
            channel = bot.get_channel(alert["channel"])
            if not channel or not NEWS_KEY:
                continue
            url = f"https://newsapi.org/v2/everything?q={alert['sujet']}&language=fr&pageSize=1&sortBy=publishedAt&apiKey={NEWS_KEY}"
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as resp:
                        data = await resp.json()
                articles = [a for a in data.get("articles", []) if a.get("title") and a["title"] != "[Removed]"]
                if articles:
                    a = articles[0]
                    embed = discord.Embed(title=f"🔔 Alerte : {alert['sujet']}", description=a.get("title"), url=a.get("url", ""), color=discord.Color.red())
                    if a.get("urlToImage"):
                        embed.set_image(url=a["urlToImage"])
                    await channel.send(embed=embed)
            except:
                pass

# =========================================================
# 🎮 DIVERTISSEMENT
# =========================================================
@bot.command(name="blague")
@anti_spam()
async def joke(ctx):
    """!v2 blague"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("https://v2.jokeapi.dev/joke/Any?lang=fr&blacklistFlags=nsfw,racist") as resp:
                data = await resp.json()
        if data.get("type") == "single":
            return await ctx.send(f"😂 {data['joke']}")
        else:
            return await ctx.send(f"😂 {data['setup']}\n||{data['delivery']}||")
    except:
        pass
    response = await ask_ai("Raconte une blague courte et drôle en français.")
    await ctx.send(f"😂 {response}")

@bot.command(name="horoscope")
@anti_spam()
async def horoscope(ctx, *, signe: str):
    """!v2 horoscope <signe>"""
    response = await ask_ai(f"Donne l'horoscope du jour pour {signe} en français, 3 phrases max, sois créatif.")
    embed = discord.Embed(title=f"⭐ Horoscope {signe.capitalize()}", description=response, color=discord.Color.gold())
    await ctx.send(embed=embed)

@bot.command(name="meteo")
async def weather(ctx, *, ville: str):
    """!v2 meteo <ville>"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"https://wttr.in/{ville}?format=j1") as resp:
                data = await resp.json()
        c = data["current_condition"][0]
        embed = discord.Embed(title=f"🌤 Météo à {ville}", color=discord.Color.blue())
        embed.add_field(name="🌡 Température", value=f"{c['temp_C']}°C (ressenti {c['FeelsLikeC']}°C)", inline=True)
        embed.add_field(name="☁️ Ciel", value=c["weatherDesc"][0]["value"], inline=True)
        embed.add_field(name="💧 Humidité", value=f"{c['humidity']}%", inline=True)
        embed.add_field(name="💨 Vent", value=f"{c['windspeedKmph']} km/h", inline=True)
        await ctx.send(embed=embed)
    except Exception as e:
        await ctx.send(f"⚠️ Ville introuvable : {e}")

@bot.command(name="quiz")
@anti_spam()
async def quiz(ctx, *, sujet: str = "culture générale"):
    """!v2 quiz <sujet>"""
    response = await ask_ai(f"Pose une question de quiz sur {sujet} avec 4 choix A B C D. Indique la bonne réponse à la fin entre crochets [X].")
    await ctx.send(f"🧠 **QUIZ : {sujet}**\n{response}")

@bot.command(name="devinette")
@anti_spam()
async def riddle(ctx):
    """!v2 devinette"""
    response = await ask_ai("Donne une devinette en français avec la réponse cachée en spoiler ||réponse||")
    await ctx.send(f"🤔 {response}")

@bot.command(name="pendu")
@anti_spam()
async def hangman(ctx):
    """!v2 pendu — partie de pendu"""
    mot_raw = await ask_ai("Donne un seul mot en français, juste le mot sans explication, entre 5 et 8 lettres.")
    mot = mot_raw.strip().lower().split()[0]
    lettres_trouvees = set()
    erreurs = 0
    max_erreurs = 6
    pendus = ["😊", "😐", "😟", "😰", "😱", "💀"]

    def afficher():
        return " ".join([c if c in lettres_trouvees else "_" for c in mot])

    await ctx.send(f"🎮 **Pendu !** Mot : `{afficher()}` ({len(mot)} lettres)\nRéponds avec une lettre !")

    def check(m):
        return m.author == ctx.author and m.channel == ctx.channel and len(m.content) == 1 and m.content.isalpha()

    while erreurs < max_erreurs:
        try:
            reponse = await bot.wait_for("message", check=check, timeout=30)
        except asyncio.TimeoutError:
            return await ctx.send(f"⏰ Temps écoulé ! Le mot était **{mot}**.")
        lettre = reponse.content.lower()
        if lettre in lettres_trouvees:
            await ctx.send("⚠️ Lettre déjà proposée !")
            continue
        if lettre in mot:
            lettres_trouvees.add(lettre)
            await reponse.add_reaction("✅")
        else:
            erreurs += 1
            await reponse.add_reaction("❌")
        affichage = afficher()
        if "_" not in affichage:
            return await ctx.send(f"🎉 Bravo ! Le mot était **{mot}** !")
        await ctx.send(f"`{affichage}` | {pendus[erreurs-1]} Erreurs : {erreurs}/{max_erreurs}")
    await ctx.send(f"💀 Perdu ! Le mot était **{mot}**.")

# =========================================================
# 📊 UTILITAIRES
# =========================================================
@bot.command(name="sondage")
async def poll(ctx, question: str, *options):
    """!v2 sondage \"question\" \"opt1\" \"opt2\""""
    if len(options) < 2:
        return await ctx.send("❌ Min 2 options. Ex : `!v2 sondage \"Pizza ou burger ?\" \"Pizza\" \"Burger\"`")
    emojis = ["1️⃣","2️⃣","3️⃣","4️⃣","5️⃣","6️⃣","7️⃣","8️⃣","9️⃣","🔟"]
    description = "\n".join([f"{emojis[i]} {opt}" for i, opt in enumerate(options[:10])])
    embed = discord.Embed(title=f"📊 {question}", description=description, color=discord.Color.green())
    embed.set_footer(text=f"Sondage par {ctx.author.display_name}")
    msg = await ctx.send(embed=embed)
    for i in range(min(len(options), 10)):
        await msg.add_reaction(emojis[i])

@bot.command(name="rappel")
async def reminder_cmd(ctx, minutes: int, *, message: str):
    """!v2 rappel <minutes> <message>"""
    await ctx.send(f"⏰ Rappel dans **{minutes} min** : {message}")
    reminders.append({"channel": ctx.channel.id, "user": ctx.author.id, "message": message, "time": asyncio.get_event_loop().time() + minutes * 60})

@tasks.loop(seconds=30)
async def check_reminders():
    now = asyncio.get_event_loop().time()
    to_remove = []
    for r in reminders:
        if now >= r["time"]:
            ch = bot.get_channel(r["channel"])
            if ch:
                await ch.send(f"⏰ <@{r['user']}> Rappel : **{r['message']}**")
            to_remove.append(r)
    for r in to_remove:
        reminders.remove(r)

@bot.command(name="stats")
async def server_stats(ctx):
    """!v2 stats — statistiques du serveur"""
    guild = ctx.guild
    bots = sum(1 for m in guild.members if m.bot)
    embed = discord.Embed(title=f"📊 Stats de {guild.name}", color=discord.Color.blue())
    if guild.icon: embed.set_thumbnail(url=guild.icon.url)
    embed.add_field(name="👥 Membres", value=str(guild.member_count), inline=True)
    embed.add_field(name="👤 Humains", value=str(guild.member_count - bots), inline=True)
    embed.add_field(name="🤖 Bots", value=str(bots), inline=True)
    embed.add_field(name="💬 Salons texte", value=str(len(guild.text_channels)), inline=True)
    embed.add_field(name="🔊 Salons vocaux", value=str(len(guild.voice_channels)), inline=True)
    embed.add_field(name="🎭 Rôles", value=str(len(guild.roles)), inline=True)
    embed.add_field(name="📅 Créé le", value=guild.created_at.strftime("%d/%m/%Y"), inline=True)
    embed.set_footer(text=f"Propriétaire : {guild.owner}")
    await ctx.send(embed=embed)

@bot.command(name="addcmd")
async def add_custom_cmd(ctx, nom: str, *, reponse: str):
    """!v2 addcmd <nom> <réponse>"""
    if ctx.guild.id not in custom_commands:
        custom_commands[ctx.guild.id] = {}
    custom_commands[ctx.guild.id][nom] = reponse
    await ctx.send(f"✅ Commande **{nom}** ajoutée !")

@bot.command(name="delcmd")
async def del_custom_cmd(ctx, nom: str):
    """!v2 delcmd <nom>"""
    if ctx.guild.id in custom_commands and nom in custom_commands[ctx.guild.id]:
        del custom_commands[ctx.guild.id][nom]
        await ctx.send(f"✅ Commande **{nom}** supprimée !")
    else:
        await ctx.send("❌ Commande introuvable.")

@bot.command(name="filtre")
async def add_filter(ctx, *, mots: str):
    """!v2 filtre <mot1 mot2 ...>"""
    if ctx.guild.id not in word_filters:
        word_filters[ctx.guild.id] = []
    nouveaux = mots.split()
    word_filters[ctx.guild.id].extend(nouveaux)
    await ctx.send(f"✅ {len(nouveaux)} mot(s) filtré(s).")

@bot.command(name="filtreoff")
async def remove_filter(ctx):
    """!v2 filtreoff"""
    word_filters.pop(ctx.guild.id, None)
    await ctx.send("✅ Filtre désactivé.")

# =========================================================
# 🎨 IMAGES
# =========================================================
@bot.command(name="img")
@anti_spam()
async def image_search(ctx, *, sujet: str):
    """!v2 img <sujet> — images Unsplash"""
    if not UNSPLASH_KEY:
        return await ctx.send("❌ UNSPLASH_API_KEY manquante.")
    await ctx.send(f"🔍 Recherche : **{sujet}**...")
    url = f"https://api.unsplash.com/search/photos?query={sujet}&per_page=3"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers={"Authorization": f"Client-ID {UNSPLASH_KEY}"}) as resp:
                data = await resp.json()
        results = data.get("results", [])
        if not results:
            return await ctx.send("❌ Aucune image.")
        for r in results:
            embed = discord.Embed(title=f"📸 {sujet}", color=discord.Color.green())
            embed.set_image(url=r["urls"]["regular"])
            embed.set_footer(text=f"Photo par {r.get('user', {}).get('name', '?')} — Unsplash")
            await ctx.send(embed=embed)
            await asyncio.sleep(0.5)
    except Exception as e:
        await ctx.send(f"⚠️ {e}")

@bot.command(name="imagine")
@anti_spam()
async def generate_image(ctx, *, prompt: str):
    """!v2 imagine <description> — génère une image IA (gratuit)"""
    await ctx.send(f"🎨 Génération : **{prompt}**...")
    encoded = prompt.replace(" ", "%20")
    url = f"https://image.pollinations.ai/prompt/{encoded}?width=512&height=512&nologo=true"
    embed = discord.Embed(title=f"🎨 {prompt}", color=discord.Color.purple())
    embed.set_image(url=url)
    embed.set_footer(text="Généré par Pollinations AI — Gratuit")
    await ctx.send(embed=embed)

@bot.command(name="imagine2")
@anti_spam()
async def generate_image_hf(ctx, *, prompt: str):
    """!v2 imagine2 <description> — génère via Hugging Face"""
    if not HF_KEY:
        return await ctx.send("❌ HF_API_KEY manquante.")
    await ctx.send(f"🎨 Génération HF : **{prompt}**...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0",
                headers={"Authorization": f"Bearer {HF_KEY}"},
                json={"inputs": prompt}
            ) as resp:
                if resp.status == 200:
                    image_bytes = await resp.read()
                    import io
                    file = discord.File(io.BytesIO(image_bytes), filename="image.png")
                    return await ctx.send(f"🎨 **{prompt}**", file=file)
    except Exception as e:
        pass
    await ctx.send("⚠️ Génération échouée, essaie `!v2 imagine` à la place.")

@bot.command(name="meme")
async def meme(ctx):
    """!v2 meme — mème aléatoire"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("https://meme-api.com/gimme") as resp:
                data = await resp.json()
        embed = discord.Embed(title=data.get("title", "Mème"), url=data.get("postLink", ""), color=discord.Color.orange())
        embed.set_image(url=data.get("url", ""))
        embed.set_footer(text=f"r/{data.get('subreddit', 'memes')}")
        await ctx.send(embed=embed)
    except Exception as e:
        await ctx.send(f"⚠️ {e}")

# =========================================================
# ❓ AIDE
# =========================================================
@bot.command(name="aide")
async def help_cmd(ctx):
    """!v2 aide"""
    embed = discord.Embed(title="🐾 IRIS V2 — Toutes les commandes", color=discord.Color.blurple())
    embed.add_field(name="🤖 IA & Chat", value="`ask` `reset` `resume` `perso` `traduire`", inline=False)
    embed.add_field(name="🎵 Musique", value="`play` `np` `skip` `pause` `reprendre` `stop` `file` `volume` `paroles` `sauver` `playlist` `recommande`", inline=False)
    embed.add_field(name="📰 Actualités", value="`news` `newsresume` `alerte` `alerteoff`", inline=False)
    embed.add_field(name="🎮 Divertissement", value="`blague` `horoscope` `meteo` `quiz` `devinette` `pendu`", inline=False)
    embed.add_field(name="📊 Utilitaires", value="`sondage` `rappel` `stats` `addcmd` `delcmd` `filtre` `filtreoff`", inline=False)
    embed.add_field(name="🎨 Images", value="`img` `imagine` `imagine2` `meme`", inline=False)
    embed.add_field(name="⚡ Anti-spam", value="Max 5 requêtes/minute par utilisateur", inline=False)
    embed.set_footer(text="Préfixe : !v2 | Ex: !v2 ask Bonjour !")
    await ctx.send(embed=embed)

# =========================================================
# RUNNER
# =========================================================
if __name__ == "__main__":
    if not TOKEN:
        logger.error("❌ DISCORD_TOKEN_V2 manquant !")
    else:
        bot.run(TOKEN)
