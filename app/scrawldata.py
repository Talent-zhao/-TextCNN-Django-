# =======================
# 新爬虫
# =======================


import csv
import html as ihtml
import json
import os
import random
import re
import time
import urllib.parse
import warnings

import requests
from lxml import etree
try:
    from playwright.sync_api import sync_playwright
except Exception:
    sync_playwright = None

warnings.filterwarnings('ignore')


# 全局 Session
session = requests.Session()

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/143.0.0.0 Safari/537.36",
    "Referer": "https://tieba.baidu.com/",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}

# Cookie 来源：
# 1) 优先读取环境变量 TIEBA_COOKIE（推荐）
# 2) 未设置时回退到本地默认值（便于开发环境快速运行）
COOKIE_STR_DEFAULT = """
PSTM=1760363733;
BIDUPSID=8ABF946C43C1F40953CC909FF683B161;
BAIDUID=6EC5CFDBD9A4D14C1A160FB798406A79:FG=1;
BDUSS=kt0cDlzVkpPYzV4NTRLQ2NOZWs4N0hDS3ZCOWVxc3l3VzNITEFZM1BPOTY0cDlwRVFBQUFBJCQAAAAAAAAAAAEAAACOT1xxaW9mZ2huAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAHpVeGl6VXhpN;
BDUSS_BFESS=kt0cDlzVkpPYzV4NTRLQ2NOZWs4N0hDS3ZCOWVxc3l3VzNITEFZM1BPOTY0cDlwRVFBQUFBJCQAAAAAAAAAAAEAAACOT1xxaW9mZ2huAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAHpVeGl6VXhpN;
BAIDUID_BFESS=6EC5CFDBD9A4D14C1A160FB798406A79:FG=1;
ZFY=dMuzfye0bt3LrT7cjYdZ0HVIGjBiZE0cONW62Q9br84:C;
H_PS_PSSID=60275_63141_67862_67887_67965_68050_67984_68002_68143_68149_68146_68139_68165_68196_68229_68232_68261_68263_68308_68350_68336_68372_68435_68451_68463_68543_68536_68550_68557_68502_68517_68594_68617_68622_68608_68600_68682_68671_68699_68742_68752_68544_68731;
H_WISE_SIDS=60275_63141_67862_67887_67965_68050_67984_68002_68143_68149_68146_68139_68165_68196_68229_68232_68261_68263_68308_68350_68336_68372_68435_68451_68463_68543_68536_68550_68557_68502_68517_68594_68617_68622_68608_68600_68682_68671_68699_68742_68752_68544_68731;
STOKEN=e3c0192b7b880ff89830585081c5583afa627584194789cef1efc1f1df2eb851;
USER_JUMP=-1;
BAIDU_WISE_UID=wapp_1774956951999_681;
__bid_n=19d43adbeb745128dcfc5e;
TIEBAUID=3815652cd3b9bf80e80d7d36;
ab_sr=1.0.1_OWQ3YWZlNzcwOGVhZWVmNWExYWQwZTY2ZTdiYzNhZWVkYmUyMDRiNTM5YTU2N2FjZGViNjgzNTBiYzE4NjRhYmQyMDdmM2ZiOWNmMTgzM2VlYzZjOWRiZTZlNDBjNzRkZDRjODhiNjFiZWNkMTI4NTIxNGJkYTM0OTRmNDM0NzAwNTg0ZTA4MzhiNzViMjcyM2NmYzJjY2MyYzYyZTdlMGFlZWVjNTAzZmRiMTY0MWE2ZGIwMDUyNWYxYTBmNjdj;
"""
COOKIE_STR = (os.environ.get("TIEBA_COOKIE") or "").strip() or COOKIE_STR_DEFAULT.strip()

def load_cookies(session, cookie_str):
    for kv in cookie_str.split(";"):
        if "=" in kv:
            k, v = kv.strip().split("=", 1)
            session.cookies.set(k, v)


def _parse_cookie_pairs(cookie_str):
    out = []
    for kv in (cookie_str or "").split(";"):
        if "=" not in kv:
            continue
        k, v = kv.strip().split("=", 1)
        if not k:
            continue
        out.append((k.strip(), v.strip()))
    return out


def _build_playwright_cookies(cookie_str):
    cookies = []
    for k, v in _parse_cookie_pairs(cookie_str):
        cookies.append(
            {
                "name": k,
                "value": v,
                "domain": ".baidu.com",
                "path": "/",
                "httpOnly": False,
                "secure": True,
            }
        )
    return cookies

if os.environ.get("TIEBA_COOKIE"):
    print("Tieba crawler: using cookie from env TIEBA_COOKIE")
else:
    print("Tieba crawler: env TIEBA_COOKIE not set, using fallback cookie in code")

load_cookies(session, COOKIE_STR)
session.headers.update(HEADERS)


# 获取页面 HTML
def get_page_html(url, timeout=10):
    resp = session.get(url, timeout=timeout)
    resp.raise_for_status()
    resp.encoding = "utf-8"
    text = resp.text or ""
    time.sleep(random.uniform(0.8, 1.8))
    return text


# 获取最大页数
def get_max_page(tree):
    page = tree.xpath('//li[@class="l_reply_num"]/span/text()')
    if page:
        try:
            return int(page[-1])
        except:
            pass
    return 1


def _looks_blocked(html_text):
    bad_keys = (
        "请输入验证码",
        "访问过于频繁",
        "百度安全验证",
        "请先登录",
        "内容不存在",
    )
    t = html_text or ""
    return any(k in t for k in bad_keys)


def _debug_dump_html(tid, page, html_text):
    os.makedirs("data", exist_ok=True)
    fp = os.path.join("data", f"debug_tieba_{tid}_p{page}.html")
    with open(fp, "w", encoding="utf-8") as f:
        f.write(html_text or "")
    return fp


def _safe_json_loads(raw):
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def _extract_by_data_field(tree):
    """
    首选：从 data-field JSON 取结构化字段，再补抓正文节点。
    适配贴吧页面 class 变化。
    """
    out = []
    seen = set()
    nodes = tree.xpath('//*[@data-field]')
    for nd in nodes:
        raw = nd.attrib.get("data-field") or ""
        raw = ihtml.unescape(raw).strip()
        j = _safe_json_loads(raw)
        if not isinstance(j, dict):
            continue
        content = j.get("content") if isinstance(j.get("content"), dict) else {}
        author = j.get("author") if isinstance(j.get("author"), dict) else {}
        post_id = content.get("post_id")
        if post_id is None:
            continue
        try:
            post_id = int(post_id)
        except Exception:
            continue
        if post_id in seen:
            continue
        seen.add(post_id)

        floor = str(content.get("post_no") or "")
        post_time = str(content.get("date") or "")
        name = str(author.get("user_name") or author.get("name_show") or "匿名用户")
        user_url = str(author.get("user_url") or "")
        if user_url and user_url.startswith("/"):
            user_url = "https://tieba.baidu.com" + user_url

        portrait = str(author.get("portrait") or "")
        img = ("https://himg.bdimg.com/sys/portrait/item/" + portrait) if portrait else ""
        level = str(author.get("level_name") or author.get("level_id") or "")
        reply = str(content.get("comment_num") or "0")
        ip = str(content.get("ip_address") or "")

        # 正文优先通过 post_content_{post_id} 抓
        txt_nodes = tree.xpath(f'//div[contains(@id,"post_content_{post_id}")]//text()')
        text = "".join([x.strip() for x in txt_nodes if x and x.strip()]).strip()
        if not text:
            txt_nodes = nd.xpath('.//div[contains(@id,"post_content_")]//text()')
            text = "".join([x.strip() for x in txt_nodes if x and x.strip()]).strip()
        if not text:
            continue

        out.append(
            {
                "time": post_time,
                "floor": floor,
                "name": name,
                "user_url": user_url,
                "img": img,
                "level": level,
                "content": text,
                "reply": reply,
                "ip": ip,
            }
        )
    return out


def _extract_by_legacy_xpath(tree):
    """
    兜底：沿用旧 XPath 规则，兼容部分老页面。
    """
    out = []
    posts = tree.xpath('//div[contains(@class,"l_post")]')
    for post in posts:
        tail = post.xpath('.//span[@class="tail-info"]/text()')
        floor = tail[-2] if len(tail) >= 2 else ""
        time1 = tail[-1] if tail else ""
        name = post.xpath('.//li[@class="d_name"]/a/text()')
        name = name[0] if name else "匿名用户"
        user_url = post.xpath('.//li[@class="d_name"]/a/@href')
        user_url = "https://tieba.baidu.com" + user_url[0] if user_url else ""
        img = post.xpath('.//li[@class="icon"]//img/@src')
        img = img[0] if img else ""
        level = post.xpath('.//div[@class="d_badge_lv"]/text()')
        level = level[0] if level else ""
        content = post.xpath('.//div[contains(@id,"post_content")]//text()')
        content = "".join([x.strip() for x in content if x and x.strip()]).strip()
        ip = post.xpath('.//span[contains(text(),"IP")]/text()')
        ip = ip[0] if ip else ""
        reply = post.xpath('.//a[@class="p_reply"]/text()')
        reply = reply[0] if reply else "0"
        if not content:
            continue
        out.append(
            {
                "time": time1,
                "floor": floor,
                "name": name,
                "user_url": user_url,
                "img": img,
                "level": level,
                "content": content,
                "reply": reply,
                "ip": ip,
            }
        )
    return out


def _append_rows_to_csv(file_path, columns, rows):
    file_exists = os.path.exists(file_path)
    with open(file_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(columns)
        for r in rows:
            writer.writerow([r.get(col, "") for col in columns])


def _pb_content_to_text(content):
    """将 getPbData 的 content 列表转为纯文本。"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if isinstance(block, dict):
            t = block.get("text") or ""
            if t:
                t = re.sub(r"<br\s*/?>", "\n", t, flags=re.I)
                parts.append(t)
        elif isinstance(block, str) and block.strip():
            parts.append(block)
    return "".join(parts).strip()


def _rows_from_pbdata_payload(payload):
    """从 getPbData 的 JSON 根对象解析楼层行；失败返回空列表。"""
    if not isinstance(payload, dict):
        return []
    if payload.get("errno") not in (0, "0"):
        return []
    data = payload.get("data")
    if not isinstance(data, dict):
        return []
    post_list = data.get("post_list")
    if not isinstance(post_list, list):
        return []
    out = []
    for post in post_list:
        if not isinstance(post, dict):
            continue
        pid = post.get("id")
        if pid is None:
            continue
        author = post.get("author") if isinstance(post.get("author"), dict) else {}
        name = str(
            author.get("name_show")
            or author.get("show_nickname")
            or author.get("name")
            or "匿名用户"
        )
        un = author.get("name") or author.get("name_show") or ""
        user_url = ""
        if un:
            user_url = "https://tieba.baidu.com/home/main?un=" + urllib.parse.quote(
                str(un), safe=""
            )
        portrait = str(author.get("portrait") or "")
        img = (
            "https://himg.bdimg.com/sys/portrait/item/" + portrait
            if portrait
            else ""
        )
        floor = str(post.get("floor") or "")
        ts = post.get("time")
        try:
            ts_i = int(ts)
            post_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts_i))
        except Exception:
            post_time = str(ts or "")
        text = _pb_content_to_text(post.get("content"))
        if not text:
            continue
        out.append(
            {
                "time": post_time,
                "floor": floor,
                "name": name,
                "user_url": user_url,
                "img": img,
                "level": "",
                "content": text,
                "reply": str(post.get("reply_num") if post.get("reply_num") is not None else "0"),
                "ip": "",
            }
        )
    return out


def _fetch_pbdata_json(tid, pn):
    url = f"https://tieba.baidu.com/mg/p/getPbData?kz={tid}&pn={pn}"
    r = session.get(url, timeout=25)
    r.raise_for_status()
    return r.json()


def _scrawl_tieba_pbdata(tid):
    """
    优先方案：贴吧移动端 JSON 接口 getPbData，不依赖 PC 页 CSR 渲染。
    需有效 Cookie（与 requests session 一致）。
    """
    print(f"开始抓取帖子 {tid}（getPbData API）")
    os.makedirs("data", exist_ok=True)
    file_path = f"data/{tid}.csv"
    columns = ["time", "floor", "name", "user_url", "img", "level", "content", "reply", "ip"]

    first = _fetch_pbdata_json(tid, 1)
    if first.get("errno") not in (0, "0"):
        raise RuntimeError(first.get("errmsg") or f"getPbData errno={first.get('errno')}")

    data = first.get("data") or {}
    page_info = data.get("page") or {}
    try:
        page_max = int(page_info.get("total_page") or 1)
    except Exception:
        page_max = 1
    if page_max < 1:
        page_max = 1

    print(f"帖子 {tid} 共 {page_max} 页（API）")

    all_rows = []
    seen = set()

    for pno in range(1, page_max + 1):
        print(f"正在抓取第 {pno} 页 / 共 {page_max} 页（getPbData）")
        payload = first if pno == 1 else _fetch_pbdata_json(tid, pno)
        rows = _rows_from_pbdata_payload(payload)
        print(f"第 {pno} 页解析到 {len(rows)} 条")
        if not rows and pno == 1:
            break
        for r in rows:
            key = (
                str(r.get("time", "")),
                str(r.get("floor", "")),
                str(r.get("user_url", "")),
                str(r.get("content", "")),
            )
            if key in seen:
                continue
            seen.add(key)
            all_rows.append(r)
        time.sleep(random.uniform(0.5, 1.2))

    if not all_rows:
        raise RuntimeError("getPbData 未解析到任何楼层（请检查 Cookie 或 tid）")

    _append_rows_to_csv(file_path, columns, all_rows)
    print(f"getPbData 抓取完成，共写入 {len(all_rows)} 条")
    return len(all_rows)


def _extract_rows_from_html(tid, page, html_text):
    if _looks_blocked(html_text):
        fp = _debug_dump_html(tid, page, html_text)
        raise RuntimeError(f"贴吧返回疑似反爬/登录校验页面，已保存调试文件：{fp}")
    tree = etree.HTML(html_text)
    rows = _extract_by_data_field(tree)
    if not rows:
        rows = _extract_by_legacy_xpath(tree)
    return rows, tree


def _scrawl_tieba_playwright(tid):
    if sync_playwright is None:
        raise RuntimeError("未安装 playwright，请先执行: pip install playwright && playwright install chromium")

    print(f"开始抓取帖子 {tid}（Playwright）")
    os.makedirs("data", exist_ok=True)
    file_path = f"data/{tid}.csv"
    columns = ["time", "floor", "name", "user_url", "img", "level", "content", "reply", "ip"]

    all_rows = []
    seen = set()

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=HEADERS.get("User-Agent"),
            locale="zh-CN",
            viewport={"width": 1366, "height": 900},
        )
        ck = _build_playwright_cookies(COOKIE_STR)
        if ck:
            context.add_cookies(ck)
        page = context.new_page()
        page.set_extra_http_headers({"Referer": "https://tieba.baidu.com/"})

        first_url = f"https://tieba.baidu.com/p/{tid}?pn=1"
        page.goto(first_url, wait_until="domcontentloaded", timeout=45000)
        # PC 贴吧为 CSR，短等待往往拿不到楼层节点；滚动促使懒加载并完成渲染
        for _ in range(6):
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            page.wait_for_timeout(1200)
        try:
            page.wait_for_selector("[data-field]", timeout=20000)
        except Exception:
            pass
        page.wait_for_timeout(800)
        html = page.content()
        rows, tree = _extract_rows_from_html(tid, 1, html)
        page_max = get_max_page(tree)
        if page_max < 1:
            page_max = 1
        print(f"帖子 {tid} 共 {page_max} 页")

        for pno in range(1, page_max + 1):
            print(f"正在抓取第 {pno} 页 / 共 {page_max} 页（Playwright）")
            url = f"https://tieba.baidu.com/p/{tid}?pn={pno}"
            page.goto(url, wait_until="domcontentloaded", timeout=45000)
            for _ in range(5):
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                page.wait_for_timeout(1000)
            try:
                page.wait_for_selector("[data-field]", timeout=15000)
            except Exception:
                pass
            page.wait_for_timeout(600)
            html = page.content()
            rows, _ = _extract_rows_from_html(tid, pno, html)
            print(f"第 {pno} 页解析到 {len(rows)} 条")
            if not rows:
                fp = _debug_dump_html(tid, pno, html)
                print(f"未检测到帖子节点，终止分页。调试文件：{fp}")
                break
            for r in rows:
                key = (
                    str(r.get("time", "")),
                    str(r.get("floor", "")),
                    str(r.get("user_url", "")),
                    str(r.get("content", "")),
                )
                if key in seen:
                    continue
                seen.add(key)
                all_rows.append(r)
            time.sleep(random.uniform(0.6, 1.4))

        context.close()
        browser.close()

    _append_rows_to_csv(file_path, columns, all_rows)
    print(f"Playwright 抓取完成，共写入 {len(all_rows)} 条")
    return len(all_rows)


def _scrawl_tieba_requests(tid):
    print(f"开始抓取帖子 {tid}（requests 兜底）")

    os.makedirs("data", exist_ok=True)
    file_path = f"data/{tid}.csv"

    columns = [
        "time", "floor", "name", "user_url",
        "img", "level", "content", "reply", "ip"
    ]

    file_exists = os.path.exists(file_path)
    file = open(file_path, "a", newline="", encoding="utf-8-sig")
    writer = csv.writer(file)

    if not file_exists:
        writer.writerow(columns)

    total_written = 0


    first_url = f"https://tieba.baidu.com/p/{tid}?pn=1"
    html = get_page_html(first_url)
    if _looks_blocked(html):
        fp = _debug_dump_html(tid, 1, html)
        file.close()
        raise RuntimeError(f"贴吧返回疑似反爬/登录校验页面，已保存调试文件：{fp}")
    tree = etree.HTML(html)

    page_max = get_max_page(tree)
    print(f"帖子 {tid} 共 {page_max} 页")

    for page in range(1, page_max + 1):
        print(f"正在抓取第 {page} 页 / 共 {page_max} 页")
        url = f"https://tieba.baidu.com/p/{tid}?pn={page}"
        html = get_page_html(url)
        if _looks_blocked(html):
            fp = _debug_dump_html(tid, page, html)
            print(f"检测到疑似反爬页面，终止抓取。调试文件：{fp}")
            break
        tree = etree.HTML(html)

        rows = _extract_by_data_field(tree)
        if not rows:
            rows = _extract_by_legacy_xpath(tree)

        print(f"第 {page} 页解析到 {len(rows)} 条")
        if not rows:
            fp = _debug_dump_html(tid, page, html)
            print(f"未检测到帖子节点，终止分页。调试文件：{fp}")
            break

        for r in rows:
            writer.writerow([r[col] for col in columns])
            total_written += 1

        time.sleep(2)

    file.close()
    print(f"requests 抓取完成，共写入 {total_written} 条")
    return total_written


# 主爬虫函数
def scrawl_tieba(tid):
    # 1) getPbData：结构化 JSON，不依赖 PC 页渲染（推荐）
    try:
        return _scrawl_tieba_pbdata(tid)
    except Exception as e:
        print(f"getPbData 抓取失败，尝试 Playwright。原因：{e}")
    # 2) Playwright：CSR 页面需滚动与等待选择器
    try:
        return _scrawl_tieba_playwright(tid)
    except Exception as e:
        print(f"Playwright 抓取失败，回退 requests HTML。原因：{e}")
        return _scrawl_tieba_requests(tid)


if __name__ == "__main__":
    scrawl_tieba("10337649644")
