import os
import json
import logging
import hashlib
import requests
import time
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI
import markdown

# =============================================================================
#  CONFIGURATION
# =============================================================================
load_dotenv()
LOG_FILE  = Path("logs/agent_eco.log")
SEEN_FILE = Path("seen_eco_articles.json")
Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# Max articles to enrich with body excerpt
ENRICH_MAX = 15
# Max articles sent to DeepSeek in one call
DEEPSEEK_MAX_ARTICLES = 50

# =============================================================================
#  ALERT THRESHOLDS — used in system prompt and signal classification
# =============================================================================
THRESHOLDS = {
    "pmi_crisis":          48.0,   # PMI below this = contraction signal
    "pmi_boom":            52.0,   # PMI above this = expansion signal
    "steel_yoy_alert":     10.0,   # Steel price YoY % change → margin alert
    "aluminium_yoy_alert": 10.0,
    "lithium_yoy_alert":   20.0,
    "air_traffic_drop":    -5.0,   # YoY % → GSE demand slowdown
    "cny_eur_move":         3.0,   # % move in CNY/EUR → competitiveness impact
}

# =============================================================================
#  KEYWORDS — economic indicators relevant to manufacturing and GSE
# =============================================================================
KEYWORDS_ECO = [
    # --- PMI / macro leading indicators ---
    "PMI", "采购经理人指数", "制造业PMI", "非制造业PMI",
    "Caixin PMI", "财新PMI", "财新制造业",
    "新订单", "出口订单", "生产指数", "就业指数",
    "manufacturing PMI", "new orders", "output index",
    "economic outlook", "business confidence", "industrial output",
    "GDP", "经济增长", "工业增加值", "固定资产投资",
    "工业生产", "制造业产出",

    # --- Raw materials — steel ---
    "钢铁", "钢价", "热轧卷板", "HRC", "冷轧", "型钢", "钢材",
    "钢铁价格", "铁矿石", "焦炭", "螺纹钢",
    "steel price", "hot rolled coil", "iron ore", "coking coal",
    "Mysteel", "钢联", "中钢协",

    # --- Raw materials — aluminium ---
    "铝", "铝价", "铝合金", "氧化铝", "电解铝",
    "aluminium price", "aluminum", "LME aluminium",

    # --- Raw materials — lithium / battery ---
    "锂", "碳酸锂", "氢氧化锂", "锂电池", "动力电池",
    "lithium price", "lithium carbonate", "battery cost", "battery cell",
    "CATL", "宁德时代", "电池成本",

    # --- Raw materials — copper / semiconductors ---
    "铜价", "铜", "copper price",
    "芯片", "半导体", "供应链", "缺芯",
    "semiconductor", "chip shortage", "supply chain",

    # --- Energy costs ---
    "电价", "工业电价", "能源成本", "天然气价格",
    "electricity price", "energy cost", "natural gas",
    "煤炭", "coal price",

    # --- Labour / manufacturing costs ---
    "人工成本", "工资", "劳动力成本", "最低工资",
    "labour cost", "wage", "manufacturing cost",
    "用工荒", "招工难",

    # --- Credit / financing ---
    "LPR", "贷款利率", "融资成本", "信贷",
    "loan prime rate", "interest rate", "credit",
    "银行贷款", "企业融资", "债券",

    # --- Infrastructure investment (demand pull) ---
    "基础设施投资", "固定资产", "专项债", "国债",
    "infrastructure investment", "special bonds", "NDRC",
    "发改委", "国家发展改革委",
    "机场建设", "航空基础设施", "新机场", "航站楼扩建",
    "airport investment", "airport construction", "runway",

    # --- Trade policy / tariffs ---
    "关税", "贸易战", "出口管制", "制裁",
    "tariff", "trade war", "export control", "sanctions",
    "中美贸易", "US-China trade", "EU tariffs",
    "一带一路", "Belt and Road", "BRI",
    "RCEP", "自贸区", "free trade",

    # --- FX / currency ---
    "人民币", "汇率", "CNY", "RMB",
    "CNY/EUR", "人民币兑欧元", "汇率波动",
    "currency", "exchange rate", "devaluation", "appreciation",

    # --- Industrial policy / electrification ---
    "新能源", "电动化", "碳中和", "碳达峰", "双碳",
    "NEV", "electric vehicle", "green manufacturing",
    "补贴", "政策支持", "产业政策",
    "柴油禁行", "diesel ban", "emission standard",
    "工业绿色转型",

    # --- Aviation demand (GSE pull-through) ---
    "民航", "航空运输", "旅客吞吐量", "货邮吞吐量",
    "航班量", "通航", "暑运", "春运",
    "air traffic", "passenger volume", "cargo volume",
    "CAAC", "民用航空", "中国民航",
    "飞机订单", "机队扩张", "aircraft order", "fleet expansion",
    "Air China", "China Eastern", "China Southern",
    "国航", "东航", "南航",

    # --- Crisis / recession signals ---
    "经济衰退", "下行压力", "产能过剩", "去库存",
    "recession", "overcapacity", "destocking", "slowdown",
    "破产", "违约", "债务危机",
    "bankruptcy", "default", "debt crisis",
    "失业率", "unemployment",

    # --- Recovery signals ---
    "经济复苏", "需求回暖", "订单增加", "产能扩张",
    "recovery", "demand pickup", "order growth", "capacity expansion",
    "消费回升", "内需", "domestic demand",

    # --- Government stimulus ---
    "刺激政策", "降准", "降息", "财政刺激",
    "stimulus", "rate cut", "reserve requirement",
    "国务院", "国常会", "经济工作会议",

    # --- Regional / manufacturing hubs ---
    "长三角", "珠三角", "成渝", "京津冀",
    "江苏", "浙江", "广东", "山东", "上海",
    "制造业集群", "产业园区",
]

# =============================================================================
#  SCRAPED SOURCES — those accessible from GitHub Actions US runners
#  Chinese sources blocked by IP are handled via Tavily below
# =============================================================================
SOURCES = [
    {
        "nom": "NBS China (国家统计局)",
        "url": "https://www.stats.gov.cn/sj/zxfb/",
        "type": "scrape_generic",
        "selector": ".list_009 a, ul li a, .con a, a",
        "base_url": "https://www.stats.gov.cn",
        "encoding": "utf-8",
    },
    {
        "nom": "NDRC China (发改委)",
        "url": "https://www.ndrc.gov.cn/xwdt/xwfb/",
        "type": "scrape_generic",
        "selector": ".list_009 a, ul li a, a",
        "base_url": "https://www.ndrc.gov.cn",
        "encoding": "utf-8",
    },
    {
        "nom": "Reuters China Economy",
        "url": "https://www.reuters.com/world/china/",
        "type": "scrape_generic",
        "selector": "a[data-testid='Heading'], h3 a, article a",
        "base_url": "https://www.reuters.com",
    },
    {
        "nom": "Bloomberg China",
        "url": "https://www.bloomberg.com/asia",
        "type": "scrape_generic",
        "selector": "h3 a, .headline a, article h2 a",
        "base_url": "https://www.bloomberg.com",
    },
    {
        "nom": "Caixin Global",
        "url": "https://www.caixinglobal.com/latest-articles/",
        "type": "scrape_generic",
        "selector": "h3 a, .article-title a, .title a, a",
        "base_url": "https://www.caixinglobal.com",
    },
    {
        "nom": "South China Morning Post — Economy",
        "url": "https://www.scmp.com/economy",
        "type": "scrape_generic",
        "selector": "h2 a, h3 a, .article__title a, a",
        "base_url": "https://www.scmp.com",
    },
    {
        "nom": "CGTN Business",
        "url": "https://www.cgtn.com/business",
        "type": "scrape_generic",
        "selector": "div.newsList a, h3 a, a",
        "base_url": "https://www.cgtn.com",
        "encoding": "utf-8",
    },
    {
        "nom": "Xinhua Finance",
        "url": "http://www.xinhuanet.com/money/",
        "type": "scrape_generic",
        "selector": "ul li a, .tit a, h3 a, a",
        "base_url": "http://www.xinhuanet.com",
        "encoding": "utf-8",
    },
    {
        "nom": "LME Metals (Metal Bulletin)",
        "url": "https://www.fastmarkets.com/metals/",
        "type": "scrape_generic",
        "selector": "h3 a, .article-title a, a",
        "base_url": "https://www.fastmarkets.com",
    },
    {
        "nom": "Steel Guru",
        "url": "https://steelguru.com/steel/china",
        "type": "scrape_generic",
        "selector": "h2 a, h3 a, .news-title a, a",
        "base_url": "https://steelguru.com",
    },
]

# =============================================================================
#  UTILITY FUNCTIONS
# =============================================================================

def normaliser_url(url, base=None):
    if not url:
        return None
    if base:
        url = urljoin(base, url)
    parsed = urlparse(url)
    url_propre = parsed._replace(query="", fragment="").geturl()
    if url_propre.endswith("/"):
        url_propre = url_propre[:-1]
    return url_propre


def charger_vus():
    if SEEN_FILE.exists():
        with open(SEEN_FILE, "r", encoding="utf-8") as f:
            try:
                return set(json.load(f))
            except Exception:
                return set()
    return set()


def sauvegarder_vus(vus):
    with open(SEEN_FILE, "w", encoding="utf-8") as f:
        json.dump(list(vus), f, ensure_ascii=False, indent=2)


def requeter_avec_retry(url, retries=3, timeout=20, **kwargs):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/126.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.8,en-US;q=0.5,en;q=0.3",
    }
    if "headers" in kwargs:
        headers.update(kwargs.pop("headers"))
    for i in range(retries):
        try:
            resp = requests.get(url, timeout=timeout, headers=headers, **kwargs)
            resp.raise_for_status()
            return resp
        except Exception as e:
            log.warning(f"Attempt {i+1}/{retries} failed for {url}: {e}")
            time.sleep(2 ** i)
    return None


# =============================================================================
#  SCRAPERS
# =============================================================================

def scrape_generic(source):
    articles = []
    resp = requeter_avec_retry(source["url"])
    if not resp:
        return articles
    try:
        encoding = source.get("encoding", "utf-8")
        soup = BeautifulSoup(resp.content, "html.parser", from_encoding=encoding)
        unique_links = {}
        for link in soup.select(source["selector"]):
            href  = link.get("href")
            titre = link.get_text(strip=True)
            if not href or not titre or len(titre) < 8:
                continue
            href = normaliser_url(href, source.get("base_url"))
            if href:
                unique_links[href] = titre
        for href, titre in list(unique_links.items())[:20]:
            articles.append({
                "source": source["nom"],
                "titre":  titre[:150],
                "lien":   href,
                "desc":   "",
                "date":   datetime.now().strftime("%Y-%m-%d"),
                "id":     hashlib.md5((titre + href).encode()).hexdigest(),
            })
    except Exception as e:
        log.warning(f"Error scraping {source['nom']}: {e}")
    log.info(f"  Scraped {source['nom']}: {len(articles)} articles")
    return articles


def collecter_tous_articles():
    tous = []
    for source in SOURCES:
        log.info(f"Collecting from: {source['nom']}")
        tous.extend(scrape_generic(source))
        time.sleep(1.0)
    log.info(f"Total raw articles collected: {len(tous)}")
    return tous


# =============================================================================
#  FILTERING
# =============================================================================

def filtrer_pertinents(articles, vus):
    nouveaux = []
    for a in articles:
        if a["id"] in vus:
            continue
        texte = (a["titre"] + " " + a.get("desc", "")).lower()
        matched = [kw for kw in KEYWORDS_ECO if kw.lower() in texte]
        if matched:
            log.info(
                f"  KEPT [{a['source']}] {a['titre'][:70]} "
                f"— matched: {matched[:3]}"
            )
            nouveaux.append(a)
        else:
            log.debug(f"  SKIP [{a['source']}] {a['titre'][:70]}")
    log.info(f"Relevant articles after filtering: {len(nouveaux)}")
    return nouveaux


# =============================================================================
#  ARTICLE ENRICHMENT
# =============================================================================

def enrichir_article(article):
    """Fetch first ~400 chars of article body for better DeepSeek context."""
    try:
        resp = requests.get(
            article["lien"],
            timeout=8,
            headers={"User-Agent": "Mozilla/5.0"},
            allow_redirects=True,
        )
        soup = BeautifulSoup(resp.content, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        text = re.sub(r"\s+", " ", soup.get_text(separator=" ", strip=True))
        article["desc"] = text[:400]
    except Exception as e:
        log.debug(f"Could not enrich {article['lien']}: {e}")
    return article


def enrichir_articles(articles):
    log.info(f"Enriching up to {ENRICH_MAX} articles with body excerpts...")
    enriched = []
    for i, a in enumerate(articles):
        if i < ENRICH_MAX and a["source"] not in (
            "Tavily Search", "DeepSeek Economic Brief"
        ):
            enriched.append(enrichir_article(a))
            time.sleep(0.4)
        else:
            enriched.append(a)
    log.info("Enrichment complete.")
    return enriched


# =============================================================================
#  TAVILY SEARCH — primary source for Chinese-language economic data
#
#  Chinese sites (NBS, Caixin Chinese, Mysteel, SMM, wind.com.cn, etc.) are
#  inaccessible from GitHub Actions US runners due to IP blocking.
#  Tavily searches from its own servers and returns content from these sources.
#
#  Strategy: 16 targeted queries covering all indicator categories.
#  Mix of Chinese and English queries to maximise coverage.
# =============================================================================

TAVILY_QUERIES = [
    # --- PMI — most critical leading indicator ---
    ("中国制造业PMI 财新 最新数据 2026", "zh"),
    ("NBS China manufacturing PMI latest 2026", "en"),
    ("中国PMI 新订单指数 出口订单 2026", "zh"),

    # --- Steel prices — key GSE margin driver ---
    ("中国钢价 热轧卷板 HRC 最新价格 2026", "zh"),
    ("China steel price HRC hot rolled coil 2026", "en"),
    ("螺纹钢价格 建材钢价 Mysteel 钢联 2026", "zh"),

    # --- Aluminium & lithium ---
    ("铝价 电解铝 LME 上海铝 最新 2026", "zh"),
    ("碳酸锂价格 锂电池成本 最新行情 2026", "zh"),

    # --- Infrastructure investment & airport projects ---
    ("中国机场建设投资 航空基础设施 发改委 2026", "zh"),
    ("China airport infrastructure investment NDRC 2026", "en"),
    ("国债 专项债 基础设施 投资计划 2026", "zh"),

    # --- Air traffic data ---
    ("中国民航 旅客吞吐量 货邮 统计 2026", "zh"),
    ("CAAC China air traffic passenger cargo statistics 2026", "en"),

    # --- Trade policy & tariffs ---
    ("中美贸易 关税 制造业 出口管制 2026", "zh"),
    ("US China trade tariffs manufacturing 2026", "en"),

    # --- Economic recovery or crisis signals ---
    ("中国经济 制造业 复苏 下行 产能 2026", "zh"),
    ("China economy manufacturing recovery slowdown 2026", "en"),

    # --- Currency ---
    ("人民币汇率 CNY EUR 美元 走势 2026", "zh"),

    # --- Industrial policy / electrification ---
    ("中国新能源 工业电动化 碳中和 补贴政策 2026", "zh"),
    ("China NEV industrial electrification policy subsidy 2026", "en"),
]


def rechercher_tavily():
    """Search for Chinese economic indicators using Tavily API.

    Tavily searches from its own servers, bypassing the IP blocks
    that prevent GitHub Actions US runners from reaching Chinese sources
    like NBS, Caixin Chinese, Mysteel, Shanghai Metals Market, etc.
    """
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        log.warning("TAVILY_API_KEY not set — skipping Tavily search.")
        return []

    found     = []
    seen_urls = set()

    for query, lang in TAVILY_QUERIES:
        try:
            resp = requests.post(
                "https://api.tavily.com/search",
                json={
                    "api_key":        api_key,
                    "query":          query,
                    "search_depth":   "basic",
                    "max_results":    5,
                    "include_answer": False,
                },
                timeout=20,
            )
            resp.raise_for_status()
            data    = resp.json()
            results = data.get("results", [])
            batch   = 0

            for r in results:
                url     = str(r.get("url", "")).strip()
                title   = str(r.get("title", "")).strip()[:150]
                content = str(r.get("content", "")).strip()[:400]

                if not url or not title or url in seen_urls:
                    continue
                seen_urls.add(url)

                found.append({
                    "source": "Tavily Search",
                    "titre":  title,
                    "lien":   url,
                    "desc":   content,
                    "date":   datetime.now().strftime("%Y-%m-%d"),
                    "id":     hashlib.md5((title + url).encode()).hexdigest(),
                })
                batch += 1

            log.info(f"  Tavily [{lang.upper()}] '{query[:50]}': {batch} results")

        except Exception as e:
            log.warning(f"Tavily search failed for '{query[:45]}': {e}")

        time.sleep(0.5)

    log.info(f"Tavily total: {len(found)} articles found")
    return found


# =============================================================================
#  DEEPSEEK ECONOMIC CONTEXT BRIEF
#
#  Runs on Mondays only. Asks DeepSeek to provide structural context on
#  China's macroeconomic environment from its training knowledge —
#  long-term trends that don't change daily but provide essential background
#  for interpreting weekly signals (e.g. steel oversupply cycle, PMI trends).
# =============================================================================

ECO_BRIEF_PROMPT = """You are a senior macroeconomic analyst specializing in China's 
industrial economy. Your client is TLD Group (Alvest subsidiary), a manufacturer of 
Ground Support Equipment (GSE) operating in China.

Based on your training knowledge, provide a structured economic context brief covering:

1. China manufacturing PMI trend (last known readings, direction)
2. Steel market in China (price trend, oversupply or shortage, key drivers)
3. Aluminium and lithium price trends (direction, key drivers)
4. China infrastructure investment outlook (airport construction pipeline)
5. China air traffic trend (passenger and cargo growth trajectory)
6. CNY/EUR exchange rate trend and manufacturing competitiveness impact
7. Key macroeconomic risks for industrial manufacturers in China (next 6 months)

For each topic return a JSON object with:
  "topic": topic name
  "status": "FAVORABLE" | "NEUTRAL" | "UNFAVORABLE" for TLD's manufacturing margins
  "trend": "IMPROVING" | "STABLE" | "DETERIORATING"
  "summary": 2-3 sentences of context
  "gse_impact": one sentence on direct impact for GSE manufacturing or demand
  "confidence": "HIGH" | "MEDIUM" | "LOW"

Return ONLY a JSON array of these objects. No markdown fences, no preamble."""


def synthese_economique_deepseek():
    """Ask DeepSeek for macroeconomic context from its training knowledge.
    Runs Mondays only to avoid redundant daily API calls.
    """
    if datetime.now().weekday() != 0:
        log.info("Economic brief: skipping (runs Mondays only)")
        return []

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        log.warning("DEEPSEEK_API_KEY not set — skipping economic brief.")
        return []

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
    log.info("Requesting economic context brief from DeepSeek...")

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": ECO_BRIEF_PROMPT}],
            max_tokens=2500,
            temperature=0.3,
        )
        text = response.choices[0].message.content or ""
        text = re.sub(r"```(?:json)?|```", "", text).strip()

        array_match = re.search(r"\[.*\]", text, re.DOTALL)
        if not array_match:
            log.warning("Economic brief: no JSON array in response")
            return []

        topics = json.loads(array_match.group(0))
        if not isinstance(topics, list):
            return []

        articles = []
        for t in topics:
            if t.get("confidence") == "LOW":
                continue
            topic   = t.get("topic", "")
            status  = t.get("status", "NEUTRAL")
            trend   = t.get("trend", "STABLE")
            summary = t.get("summary", "")
            impact  = t.get("gse_impact", "")

            desc = (
                f"Status for TLD: {status}. Trend: {trend}. "
                f"{summary} GSE impact: {impact}"
            )[:400]

            articles.append({
                "source": "DeepSeek Economic Brief",
                "titre":  f"{topic} — {status} / {trend}"[:150],
                "lien":   "#eco-brief",
                "desc":   desc,
                "date":   datetime.now().strftime("%Y-%m-%d"),
                "id":     hashlib.md5(
                    (topic + datetime.now().strftime("%Y-W%W")).encode()
                ).hexdigest(),
            })

        log.info(f"Economic brief: {len(articles)} topics injected")
        return articles

    except json.JSONDecodeError as e:
        log.warning(f"Economic brief: JSON parse error — {e}")
        return []
    except Exception as e:
        log.warning(f"Economic brief failed: {e}")
        return []


# =============================================================================
#  DEEPSEEK — STRUCTURED ANALYSIS PROMPT
# =============================================================================

SYSTEM_PROMPT = """You are a senior macroeconomic analyst advising the CFO and CEO of 
TLD Group (Alvest subsidiary), a global manufacturer and lessor of Ground Support Equipment (GSE) 
with primary manufacturing operations in Shanghai and Wuxi, China.

YOUR ROLE: translate economic signals into actionable manufacturing, financial, and 
commercial intelligence for TLD's China operations and APAC region.

ANALYSIS FRAMEWORK — interpret every signal through these lenses:

MANUFACTURING MARGINS:
- Steel, aluminium, lithium, copper price moves → direct COGS impact (quantify in %)
- Energy cost changes → factory operating cost impact
- Labour cost trends → Yangtze Delta manufacturing competitiveness
- Supply chain disruptions → production risk

GSE DEMAND SIGNALS:
- PMI manufacturing > 52 = expansion → customer capex unlock → GSE orders likely
- PMI < 48 = contraction → customers defer capex → GSE demand at risk
- Airport investment announcements → GSE procurement pipeline (quantify: 1 new terminal ≈ 50-80 GSE units)
- Air traffic growth > 5% YoY → handlers need fleet refresh → opportunity
- Air traffic drop > 5% YoY → handlers defer purchases → risk
- Airline fleet orders → GSE demand follow-through in 12-18 months

FINANCIAL / FX:
- CNY depreciation vs EUR → TLD exports from China become more competitive
- CNY appreciation → reverse; also increases EUR-denominated revenue in CNY terms
- LPR rate cuts → cheaper leasing financing for customers → GSE leasing demand up
- Credit tightening → customers delay large capex → GSE sales risk

POLICY & REGULATION:
- Government stimulus packages → accelerated airport construction → demand surge
- Diesel ban policies → electric GSE mandate → TLD electrification portfolio opportunity
- Trade tariffs on steel/aluminium → import cost impact for non-Chinese competitors
- NEV subsidies → battery cost reduction → electric GSE economics improve

IMPACT LEVELS:
- CRITICAL: Act within 48h — major threshold breach, urgent opportunity/threat
- IMPORTANT: Act this week — significant shift requiring management attention
- WATCH: Monitor — emerging trend, no immediate action
- INFO: Background context

KEY THRESHOLDS TO FLAG AS CRITICAL:
- PMI < 48 or > 52 for the first time in 3+ months
- Steel or aluminium price move > 10% in 30 days
- Lithium price move > 20% in 30 days
- Air traffic YoY drop > 5%
- CNY/EUR move > 3% in 30 days
- Major infrastructure investment announcement > RMB 50bn

OUTPUT FORMAT — use EXACTLY this structure:

For each meaningful signal:
===SIGNAL_START===
SIGNAL_ID: [number]
IMPACT: [CRITICAL | IMPORTANT | WATCH | INFO]
INDICATOR: [PMI | STEEL | ALUMINIUM | LITHIUM | COPPER | ENERGY | INFRASTRUCTURE | AIR_TRAFFIC | FX | TRADE | POLICY | OTHER]
HEADLINE: [One sharp sentence — max 15 words]
READING: [2-3 sentences: what the data shows and what is driving it]
MANUFACTURING_IMPACT: [2-3 sentences: direct impact on TLD's production costs, margins, or supply chain in China]
DEMAND_IMPACT: [2-3 sentences: impact on GSE demand — airport investment, airline fleet, handler capex]
ACTION: [1-2 sentences: specific recommended action for CFO/operations team, time-bound]
===SIGNAL_END===

After ALL signals:
===SUMMARY_START===
EXECUTIVE_SUMMARY: [5-6 sentences for board-level briefing. Overall economic picture, key numbers, net impact on TLD China, priority actions.]
MARGIN_OUTLOOK: [2-3 sentences: net manufacturing margin outlook for next 30-90 days based on today's signals]
DEMAND_OUTLOOK: [2-3 sentences: GSE demand outlook for next 90-180 days]
WATCH_1: [Most critical indicator to monitor this week with specific threshold]
WATCH_2: [Second key indicator with threshold]
WATCH_3: [Third key indicator with threshold]
MAIN_RISK: [Single biggest economic risk for TLD China operations — one sentence]
MAIN_OPPORTUNITY: [Single biggest economic opportunity — one sentence]
===SUMMARY_END===

Rules:
- English only in the output
- Always quantify when possible (%, RMB values, EUR impact, unit volumes)
- No bullet points inside field values — plain prose only
- Skip articles with zero connection to manufacturing economics or GSE demand
- Always output the SUMMARY block
- Flag threshold breaches explicitly in the HEADLINE using words like ALERT or BREACH
"""


def construire_prompt_user(articles):
    date_str = datetime.now().strftime("%d %B %Y")
    lines = [
        "CHINA ECONOMIC WATCH — TLD Group / Manufacturing & GSE",
        f"Date: {date_str}",
        f"Articles to analyze: {len(articles)}",
        "",
    ]
    for i, a in enumerate(articles, 1):
        lines.append(f"[{i}] SOURCE: {a['source']}")
        lines.append(f"    TITLE: {a['titre']}")
        lines.append(f"    URL: {a['lien']}")
        if a.get("desc"):
            lines.append(f"    EXCERPT: {a['desc'][:350]}")
        lines.append("")

    lines.append(
        "Analyze each article for economic signals relevant to TLD Group's "
        "China manufacturing operations and GSE market. "
        "Output ONLY the structured blocks defined in your instructions."
    )
    lines.append("")
    lines.append(
        "CRITICAL RULE: Any article containing specific data points — PMI readings, "
        "steel prices, aluminium prices, lithium prices, air traffic statistics, "
        "infrastructure investment amounts, interest rate changes, or FX moves — "
        "MUST generate a signal regardless of how brief the mention. "
        "Quantify every data point you find. "
        "Flag any threshold breach (PMI<48, steel>+10%, lithium>+20%, traffic<-5%) "
        "as CRITICAL impact."
    )
    return "\n".join(lines)


def analyser_avec_deepseek(articles):
    if not articles:
        log.info("No articles to analyze.")
        return ""

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY environment variable not set.")

    batch = articles[:DEEPSEEK_MAX_ARTICLES]
    if len(articles) > DEEPSEEK_MAX_ARTICLES:
        log.warning(
            f"Capped input at {DEEPSEEK_MAX_ARTICLES} articles "
            f"(had {len(articles)})."
        )

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
    log.info(f"Sending {len(batch)} articles to DeepSeek...")

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": construire_prompt_user(batch)},
            ],
            max_tokens=8192,
            temperature=0.2,
        )
        raw = response.choices[0].message.content
        log.info(f"DeepSeek response: {len(raw)} chars")
        return raw
    except Exception as e:
        log.error(f"DeepSeek API error: {e}")
        return ""


# =============================================================================
#  PARSER
# =============================================================================

def extract_field(block, field):
    pattern = rf"^{field}:\s*(.+?)(?=\n[A-Z_]{{2,}}:|$)"
    match = re.search(pattern, block, re.MULTILINE | re.DOTALL)
    return match.group(1).strip() if match else ""


def parser_analyse(raw_text):
    signals = []
    summary = {
        "executive_summary": "",
        "margin_outlook":    "",
        "demand_outlook":    "",
        "watch":             [],
        "main_risk":         "",
        "main_opportunity":  "",
    }

    if not raw_text:
        log.warning("Empty DeepSeek response — nothing to parse.")
        return signals, summary

    n_starts    = raw_text.count("===SIGNAL_START===")
    n_ends      = raw_text.count("===SIGNAL_END===")
    has_summary = "===SUMMARY_START===" in raw_text

    if n_starts != n_ends:
        log.warning(
            f"TRUNCATION DETECTED: {n_starts} starts, {n_ends} ends. "
            "Raise max_tokens or reduce input."
        )
    if n_starts > 0 and not has_summary:
        log.warning("TRUNCATION DETECTED: no SUMMARY block found.")

    for block in re.findall(
        r"===SIGNAL_START===(.*?)===SIGNAL_END===", raw_text, re.DOTALL
    ):
        impact = extract_field(block, "IMPACT").upper() or "INFO"
        if impact not in ("CRITICAL", "IMPORTANT", "WATCH", "INFO"):
            impact = "INFO"
        signals.append({
            "id":                  extract_field(block, "SIGNAL_ID"),
            "impact":              impact,
            "indicator":           extract_field(block, "INDICATOR"),
            "headline":            extract_field(block, "HEADLINE"),
            "reading":             extract_field(block, "READING"),
            "manufacturing_impact":extract_field(block, "MANUFACTURING_IMPACT"),
            "demand_impact":       extract_field(block, "DEMAND_IMPACT"),
            "action":              extract_field(block, "ACTION"),
        })

    sm = re.search(
        r"===SUMMARY_START===(.*?)===SUMMARY_END===", raw_text, re.DOTALL
    )
    if sm:
        b = sm.group(1)
        summary["executive_summary"] = extract_field(b, "EXECUTIVE_SUMMARY")
        summary["margin_outlook"]    = extract_field(b, "MARGIN_OUTLOOK")
        summary["demand_outlook"]    = extract_field(b, "DEMAND_OUTLOOK")
        summary["main_risk"]         = extract_field(b, "MAIN_RISK")
        summary["main_opportunity"]  = extract_field(b, "MAIN_OPPORTUNITY")
        summary["watch"] = [
            extract_field(b, f"WATCH_{i}")
            for i in range(1, 4)
            if extract_field(b, f"WATCH_{i}")
        ]

    log.info(
        f"Parsed: {len(signals)} signals, "
        f"summary={'yes' if summary['executive_summary'] else 'NO'}"
    )
    return signals, summary


# =============================================================================
#  HTML REPORT
# =============================================================================

IMPACT_CONFIG = {
    "CRITICAL": {"label": "Critical",  "color": "#dc2626", "bg": "#fef2f2",
                 "border": "#fecaca", "text": "#991b1b"},
    "IMPORTANT": {"label": "Important", "color": "#d97706", "bg": "#fffbeb",
                  "border": "#fde68a", "text": "#92400e"},
    "WATCH":     {"label": "Watch",     "color": "#0369a1", "bg": "#f0f9ff",
                  "border": "#bae6fd", "text": "#0c4a6e"},
    "INFO":      {"label": "Info",      "color": "#6b7280", "bg": "#f9fafb",
                  "border": "#e5e7eb", "text": "#374151"},
}

INDICATOR_ICONS = {
    "PMI":            "📊",
    "STEEL":          "🔩",
    "ALUMINIUM":      "⚙️",
    "LITHIUM":        "🔋",
    "COPPER":         "🔌",
    "ENERGY":         "⚡",
    "INFRASTRUCTURE": "🏗️",
    "AIR_TRAFFIC":    "✈️",
    "FX":             "💱",
    "TRADE":          "🌐",
    "POLICY":         "📋",
    "OTHER":          "📌",
}


def md(text):
    if not text:
        return ""
    html = markdown.markdown(text.strip(), extensions=["nl2br"])
    if html.count("<p>") == 1:
        html = re.sub(r"^<p>(.*)</p>$", r"\1", html, flags=re.DOTALL)
    return html


def trouver_article(sig, articles):
    haystack = (
        sig.get("headline", "") + " " + sig.get("reading", "")
    ).lower()
    best_article, best_score = None, 0
    for a in articles:
        candidate = (a["titre"] + " " + a.get("desc", "")).lower()
        words = [w for w in re.split(r"[\s\W]+", candidate) if len(w) >= 3]
        score = sum(1 for w in words if w in haystack)
        if score > best_score:
            best_score, best_article = score, a
    return best_article if best_score >= 1 else None


def render_signal_card(sig, articles):
    cfg  = IMPACT_CONFIG.get(sig["impact"], IMPACT_CONFIG["INFO"])
    icon = INDICATOR_ICONS.get(sig.get("indicator", "OTHER"), "📌")
    article      = trouver_article(sig, articles)
    source_block = ""
    if article and article.get("lien", "#") != "#eco-brief":
        titre_esc = (
            article["titre"]
            .replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        )
        source_block = (
            f'<div class="signal-source">'
            f'<span class="source-label">Source</span>'
            f'<a href="{article.get("lien","#")}" target="_blank" rel="noopener">'
            f"{titre_esc}</a>"
            f'<span class="source-name"> — {article["source"]}</span>'
            f"</div>"
        )

    mfg_block = ""
    if sig.get("manufacturing_impact"):
        mfg_block = f"""
    <div class="signal-section">
      <div class="signal-section-label">Manufacturing impact</div>
      <div class="signal-section-text">{md(sig['manufacturing_impact'])}</div>
    </div>"""

    demand_block = ""
    if sig.get("demand_impact"):
        demand_block = f"""
    <div class="signal-section">
      <div class="signal-section-label">GSE demand impact</div>
      <div class="signal-section-text">{md(sig['demand_impact'])}</div>
    </div>"""

    return f"""
<div class="signal-card impact-{sig['impact'].lower()}">
  <div class="signal-card-header" style="border-left:4px solid {cfg['color']};">
    <span class="signal-badge"
          style="background:{cfg['bg']};color:{cfg['text']};border:1px solid {cfg['border']};">
      {cfg['label']}
    </span>
    <span class="indicator-tag">{icon} {sig.get('indicator','')}</span>
    <h3 class="signal-headline">{md(sig['headline'])}</h3>
  </div>
  <div class="signal-body">
    <div class="signal-section">
      <div class="signal-section-label">Reading</div>
      <div class="signal-section-text">{md(sig['reading'])}</div>
    </div>
    {mfg_block}
    {demand_block}
    <div class="signal-section signal-action">
      <div class="signal-section-label">Recommended action</div>
      <div class="signal-section-text">{md(sig['action'])}</div>
    </div>
    {source_block}
  </div>
</div>"""


def generer_rapport(articles, signals, summary, truncated=False):
    now_full = datetime.now().strftime("%B %d, %Y")
    now_time = datetime.now().strftime("%H:%M")

    counts = {"CRITICAL": 0, "IMPORTANT": 0, "WATCH": 0, "INFO": 0}
    for s in signals:
        counts[s["impact"]] = counts.get(s["impact"], 0) + 1

    actionable  = [s for s in signals if s["impact"] in ("CRITICAL", "IMPORTANT", "WATCH")]
    background  = [s for s in signals if s["impact"] == "INFO"]

    signals_html = ""
    if not actionable and not background:
        signals_html = (
            '<p style="color:#6b7280;font-style:italic;padding:24px 0;">'
            "No significant economic signals identified today.</p>"
        )
    else:
        for sig in actionable:
            signals_html += render_signal_card(sig, articles)
        if background:
            info_items = "".join(
                f'<li style="font-size:13px;color:#64748b;padding:3px 0;">'
                f'{sig["headline"]}</li>'
                for sig in background
            )
            signals_html += f"""
<details style="margin-top:12px;">
  <summary style="font-size:12px;color:#94a3b8;cursor:pointer;padding:8px 4px;
                  user-select:none;list-style:none;">
    <span style="font-size:10px;background:#f1f5f9;border:1px solid #e2e8f0;
                 border-radius:20px;padding:2px 8px;color:#64748b;font-weight:600;">
      + {len(background)} background item{"s" if len(background)!=1 else ""}
    </span>
    <span style="color:#94a3b8;margin-left:8px;">— no immediate action, click to expand</span>
  </summary>
  <ul style="list-style:none;padding:12px 16px;margin-top:8px;
             background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;">
    {info_items}
  </ul>
</details>"""

    counter_html = "".join(
        f'<span class="counter-pill" '
        f'style="background:{IMPACT_CONFIG[lvl]["bg"]};'
        f'color:{IMPACT_CONFIG[lvl]["text"]};'
        f'border:1px solid {IMPACT_CONFIG[lvl]["border"]};">'
        f'{counts[lvl]} {IMPACT_CONFIG[lvl]["label"]}</span>'
        for lvl in ("CRITICAL", "IMPORTANT", "WATCH", "INFO")
        if counts[lvl] > 0
    )

    watch_html   = "".join(f"<li>{md(w)}</li>" for w in summary.get("watch", []))
    exec_html    = md(summary.get("executive_summary", ""))
    margin_html  = md(summary.get("margin_outlook", ""))
    demand_html  = md(summary.get("demand_outlook", ""))
    risk_html    = md(summary.get("main_risk", ""))
    opp_html     = md(summary.get("main_opportunity", ""))
    sources_list = "".join(f"<li>{s['nom']}</li>" for s in SOURCES)

    trunc_banner = ""
    if truncated:
        trunc_banner = """
<div style="background:#fef9c3;border:1px solid #fde047;border-radius:8px;
            padding:12px 16px;margin-bottom:24px;font-size:13px;color:#713f12;">
  <strong>Warning:</strong> DeepSeek output may have been truncated.
  Some signals could be missing.
</div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>China Economic Watch — {now_full}</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
:root{{
  --ink:#0f172a;--ink-2:#334155;--ink-3:#64748b;--ink-4:#94a3b8;
  --surface:#fff;--surface-1:#f8fafc;--border:#e2e8f0;
  --radius:8px;--radius-lg:12px;
}}
body{{font-family:'Inter',-apple-system,sans-serif;background:#f0f2f5;
      color:var(--ink);line-height:1.6;padding:32px 16px 64px}}
.wrapper{{max-width:960px;margin:0 auto}}

/* MASTHEAD */
.masthead{{background:var(--ink);border-radius:var(--radius-lg) var(--radius-lg) 0 0;padding:28px 36px 24px}}
.masthead-eyebrow{{font-family:'IBM Plex Mono',monospace;font-size:10px;letter-spacing:.12em;
                   text-transform:uppercase;color:#64748b;margin-bottom:8px}}
.masthead-title{{font-size:22px;font-weight:600;letter-spacing:-.02em;color:#fff;margin-bottom:4px}}
.masthead-subtitle{{font-size:13px;color:#475569;margin-bottom:12px}}
.masthead-meta{{display:flex;align-items:center;gap:20px;flex-wrap:wrap}}
.meta-item{{font-size:13px;color:#94a3b8;display:flex;align-items:center;gap:6px}}
.meta-item strong{{color:#e2e8f0;font-weight:500}}
.masthead-counters{{display:flex;gap:8px;flex-wrap:wrap;margin-top:16px;
                    padding-top:16px;border-top:1px solid #1e293b}}
.counter-pill{{font-size:11px;font-weight:500;padding:3px 10px;border-radius:20px;letter-spacing:.02em}}

/* CARD BODY */
.card-body{{background:var(--surface);border:1px solid var(--border);border-top:none;
            border-radius:0 0 var(--radius-lg) var(--radius-lg);padding:36px}}
.section-header{{display:flex;align-items:center;gap:10px;margin-bottom:20px;
                 padding-bottom:12px;border-bottom:1px solid var(--border)}}
.section-header h2{{font-size:13px;font-weight:600;text-transform:uppercase;
                    letter-spacing:.08em;color:var(--ink-3)}}
.section-divider{{margin:36px 0;border:none;border-top:1px solid var(--border)}}

/* EXEC SUMMARY */
.exec-panel{{background:var(--ink);border-radius:var(--radius-lg);padding:24px 28px;
             margin-bottom:24px;color:#e2e8f0;font-size:15px;line-height:1.75}}
.exec-panel-label{{font-family:'IBM Plex Mono',monospace;font-size:10px;letter-spacing:.1em;
                   text-transform:uppercase;color:#475569;margin-bottom:10px}}
.exec-panel p{{margin:0}}

/* OUTLOOK PANELS */
.outlook-grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:24px}}
.outlook-card{{border-radius:var(--radius);padding:16px 18px;border:1px solid}}
.outlook-card.margins{{background:#f0fdf4;border-color:#bbf7d0}}
.outlook-card.demand{{background:#eff6ff;border-color:#bfdbfe}}
.outlook-label{{font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.1em;
                margin-bottom:8px}}
.outlook-card.margins .outlook-label{{color:#166534}}
.outlook-card.demand .outlook-label{{color:#1e40af}}
.outlook-text{{font-size:13px;line-height:1.6}}
.outlook-card.margins .outlook-text{{color:#14532d}}
.outlook-card.demand .outlook-text{{color:#1e3a8a}}
.outlook-text p{{margin:0}}

/* SIGNAL CARDS */
.signal-card{{border:1px solid var(--border);border-radius:var(--radius-lg);
              margin-bottom:16px;overflow:hidden;transition:box-shadow .15s}}
.signal-card:hover{{box-shadow:0 4px 12px rgba(0,0,0,.06)}}
.signal-card-header{{padding:14px 20px;background:var(--surface-1);
                     display:flex;align-items:flex-start;gap:10px;flex-wrap:wrap}}
.signal-badge{{font-size:11px;font-weight:600;padding:3px 9px;border-radius:20px;
               white-space:nowrap;letter-spacing:.03em;margin-top:2px;flex-shrink:0}}
.indicator-tag{{font-size:11px;font-weight:500;color:var(--ink-3);
                background:#f1f5f9;padding:3px 8px;border-radius:6px;
                white-space:nowrap;margin-top:2px;flex-shrink:0}}
.signal-headline{{font-size:15px;font-weight:600;color:var(--ink);line-height:1.4;flex:1}}
.signal-headline p{{margin:0}}
.signal-body{{padding:20px;display:grid;gap:14px}}
.signal-section-label{{font-size:10px;font-weight:600;text-transform:uppercase;
                        letter-spacing:.1em;color:var(--ink-4);margin-bottom:4px}}
.signal-section-text{{font-size:14px;color:var(--ink-2);line-height:1.65}}
.signal-section-text p{{margin:0}}
.signal-action .signal-section-text{{color:var(--ink);font-weight:500}}
.signal-source{{padding-top:10px;border-top:1px dashed var(--border);font-size:12px;
                color:var(--ink-4);display:flex;flex-wrap:wrap;gap:4px;align-items:center}}
.source-label{{font-weight:600;text-transform:uppercase;letter-spacing:.06em;
               font-size:10px;color:var(--ink-4);margin-right:4px}}
.signal-source a{{color:#2563eb;text-decoration:none;font-weight:500}}
.signal-source a:hover{{text-decoration:underline}}
.source-name{{color:var(--ink-4)}}

/* WATCH / RISK / OPP */
.watch-panel{{background:#fffbeb;border:1px solid #fde68a;border-radius:var(--radius-lg);
              padding:18px 22px;margin-bottom:12px}}
.watch-panel-label{{font-size:11px;font-weight:600;text-transform:uppercase;
                    letter-spacing:.08em;color:#92400e;margin-bottom:10px}}
.watch-panel ol{{padding-left:20px;display:grid;gap:6px}}
.watch-panel li{{font-size:14px;color:#78350f;line-height:1.5}}
.watch-panel li p{{margin:0}}
.risk-panel{{background:#fef2f2;border:1px solid #fecaca;border-radius:var(--radius-lg);
             padding:16px 22px;margin-bottom:12px}}
.risk-panel-label{{font-size:11px;font-weight:600;text-transform:uppercase;
                   letter-spacing:.08em;color:#991b1b;margin-bottom:6px}}
.risk-panel-text{{font-size:14px;color:#7f1d1d;font-weight:500;line-height:1.6}}
.risk-panel-text p{{margin:0}}
.opp-panel{{background:#f0fdf4;border:1px solid #bbf7d0;border-radius:var(--radius-lg);
            padding:16px 22px}}
.opp-panel-label{{font-size:11px;font-weight:600;text-transform:uppercase;
                  letter-spacing:.08em;color:#166534;margin-bottom:6px}}
.opp-panel-text{{font-size:14px;color:#14532d;font-weight:500;line-height:1.6}}
.opp-panel-text p{{margin:0}}

/* SOURCES */
.sources-panel{{background:var(--surface-1);border:1px solid var(--border);
                border-radius:var(--radius);padding:14px 18px;margin-top:32px}}
.sources-panel-label{{font-size:10px;font-weight:600;text-transform:uppercase;
                       letter-spacing:.1em;color:var(--ink-4);margin-bottom:8px}}
.sources-panel ul{{list-style:none;display:flex;flex-wrap:wrap;gap:4px 0;
                   column-gap:24px;columns:2}}
.sources-panel li{{font-size:12px;color:var(--ink-3);break-inside:avoid}}
.sources-panel li::before{{content:"·";margin-right:6px;color:var(--ink-4)}}

/* FOOTER */
.page-footer{{text-align:center;font-size:11px;color:var(--ink-4);margin-top:24px;
              font-family:'IBM Plex Mono',monospace;letter-spacing:.04em}}

@media(max-width:640px){{
  body{{padding:12px 8px 48px}}
  .masthead,.card-body{{padding:20px 16px}}
  .outlook-grid{{grid-template-columns:1fr}}
  .sources-panel ul{{columns:1}}
}}
</style>
</head>
<body>
<div class="wrapper">

<div class="masthead">
  <div class="masthead-eyebrow">TLD Group · Alvest · CFO Intelligence</div>
  <div class="masthead-title">China Economic Watch</div>
  <div class="masthead-subtitle">Manufacturing margins · GSE demand · Raw materials · Infrastructure</div>
  <div class="masthead-meta">
    <div class="meta-item"><span>Date</span><strong>{now_full}</strong></div>
    <div class="meta-item"><span>Generated</span><strong>{now_time}</strong></div>
    <div class="meta-item"><span>Articles analyzed</span><strong>{len(articles)}</strong></div>
    <div class="meta-item"><span>Signals</span><strong>{len(signals)}</strong></div>
  </div>
  {f'<div class="masthead-counters">{counter_html}</div>' if counter_html else ''}
</div>

<div class="card-body">

  {trunc_banner}

  {f'<div class="exec-panel"><div class="exec-panel-label">Executive summary</div>{exec_html}</div>' if exec_html else ''}

  {f'''<div class="outlook-grid">
    <div class="outlook-card margins">
      <div class="outlook-label">📉 Manufacturing margin outlook (30-90 days)</div>
      <div class="outlook-text">{margin_html}</div>
    </div>
    <div class="outlook-card demand">
      <div class="outlook-label">📈 GSE demand outlook (90-180 days)</div>
      <div class="outlook-text">{demand_html}</div>
    </div>
  </div>''' if margin_html or demand_html else ''}

  <div class="section-header"><h2>Signals</h2></div>
  {signals_html}

  {'<hr class="section-divider">' if watch_html or risk_html or opp_html else ''}

  {f'<div class="section-header"><h2>To watch this week</h2></div><div class="watch-panel"><div class="watch-panel-label">Key indicators &amp; thresholds</div><ol>{watch_html}</ol></div>' if watch_html else ''}

  {f'<div class="risk-panel"><div class="risk-panel-label">Main risk</div><div class="risk-panel-text">{risk_html}</div></div>' if risk_html else ''}

  {f'<div class="opp-panel"><div class="opp-panel-label">Main opportunity</div><div class="opp-panel-text">{opp_html}</div></div>' if opp_html else ''}

  <div class="sources-panel">
    <div class="sources-panel-label">Monitored sources (scraped + Tavily)</div>
    <ul>
      {sources_list}
      <li>Tavily Search (NBS, Caixin ZH, Mysteel, SMM, NDRC, CAAC, Wind)</li>
    </ul>
  </div>

</div>

<div class="page-footer">
  China Economic Watch · TLD Group / Alvest · Powered by DeepSeek + Tavily · {now_full}
</div>

</div>
</body>
</html>"""


# =============================================================================
#  SAVE
# =============================================================================

def sauvegarder_rapport(rapport_html):
    dossier = Path("rapports")
    dossier.mkdir(exist_ok=True, parents=True)
    fichier = dossier / f"eco_watch_{datetime.now().strftime('%Y%m%d_%H%M')}.html"
    with open(fichier, "w", encoding="utf-8") as f:
        f.write(rapport_html)
    log.info(f"Report saved: {fichier.absolute()}")
    return fichier


# =============================================================================
#  MAIN
# =============================================================================

def executer_agent():
    log.info("=" * 60)
    log.info("Starting China Economic Watch Agent v1.0")
    log.info("=" * 60)
    try:
        vus = charger_vus()

        # 1. Scraped sources (English/international — accessible from GitHub US)
        tous_articles = collecter_tous_articles()

        # 2. Tavily search — primary channel for Chinese-language sources
        #    (NBS, Caixin ZH, Mysteel, SMM, NDRC, CAAC stats, etc.)
        #    Tavily bypasses IP blocking from GitHub Actions US runners.
        tavily_articles = rechercher_tavily()
        if tavily_articles:
            tous_articles.extend(tavily_articles)
            log.info(f"Total after Tavily: {len(tous_articles)} articles")

        # 3. DeepSeek macro context brief (Mondays only)
        eco_articles = synthese_economique_deepseek()
        if eco_articles:
            tous_articles.extend(eco_articles)
            log.info(f"Total after eco brief: {len(tous_articles)} articles")

        # 4. Filter by keywords
        articles_pertinents = filtrer_pertinents(tous_articles, vus)

        # 5. Prioritize: Tavily first, eco brief second, scraped last
        #    Ensures Chinese-language data is never dropped by the DeepSeek cap
        def source_priority(a):
            if a["source"] == "Tavily Search":
                return 0
            if a["source"] == "DeepSeek Economic Brief":
                return 1
            return 2

        articles_pertinents.sort(key=source_priority)
        log.info(
            f"After prioritization: "
            f"{sum(1 for a in articles_pertinents if a['source'] == 'Tavily Search')} Tavily, "
            f"{sum(1 for a in articles_pertinents if a['source'] == 'DeepSeek Economic Brief')} eco-brief, "
            f"{sum(1 for a in articles_pertinents if a['source'] not in ('Tavily Search','DeepSeek Economic Brief'))} scraped"
        )

        # 6. Enrich scraped articles (Tavily/eco-brief already have content)
        if articles_pertinents:
            articles_pertinents = enrichir_articles(articles_pertinents)

        # 7. Analyze with DeepSeek
        raw_analyse = (
            analyser_avec_deepseek(articles_pertinents)
            if articles_pertinents
            else ""
        )

        # 8. Save raw output for debugging
        Path("rapports").mkdir(exist_ok=True, parents=True)
        Path("rapports/debug_raw_eco.txt").write_text(
            raw_analyse or "", encoding="utf-8"
        )
        log.info("Raw DeepSeek output saved to rapports/debug_raw_eco.txt")

        # 9. Truncation detection
        n_starts  = raw_analyse.count("===SIGNAL_START===")
        n_ends    = raw_analyse.count("===SIGNAL_END===")
        has_sum   = "===SUMMARY_START===" in raw_analyse
        truncated = (n_starts != n_ends) or (n_starts > 0 and not has_sum)

        # 10. Parse
        signals, summary = parser_analyse(raw_analyse)

        # 11. Generate report
        rapport_html = generer_rapport(
            articles_pertinents, signals, summary, truncated=truncated
        )

        # 12. Save
        fichier = sauvegarder_rapport(rapport_html)
        print(f"✅ Report generated: {fichier}")

        # 13. Mark as seen
        for a in articles_pertinents:
            vus.add(a["id"])
        sauvegarder_vus(vus)
        log.info("Done.")

    except Exception as e:
        log.exception(f"Fatal error: {e}")
        raise


if __name__ == "__main__":
    executer_agent()
