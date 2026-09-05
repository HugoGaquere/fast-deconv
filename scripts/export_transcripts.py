#!/usr/bin/env python3
"""Export Claude Code transcripts for this project to an index + per-session markdown."""
import difflib, glob, json, os, sys
from collections import Counter

SRC = os.path.expanduser("~/.claude/projects/-home-hugo-Projects-fast-deconv")
OUT = sys.argv[1] if len(sys.argv) > 1 else "llm-audit"
KEEP_THINKING = "--thinking" in sys.argv
CLIP = 800


def clip(s, n=CLIP):
    s = str(s)
    return s if len(s) <= n else s[:n] + f"\n... [{len(s) - n} more chars]"


def blocks(msg):
    c = msg.get("content")
    return [{"type": "text", "text": c}] if isinstance(c, str) else (c or [])


def render_tool(name, inp):
    if name == "Bash":
        return "```bash\n%s\n```" % inp.get("command", "")
    if name == "Edit":
        old, new = inp.get("old_string", ""), inp.get("new_string", "")
        d = difflib.unified_diff(old.splitlines(), new.splitlines(), "before", "after", lineterm="", n=1)
        return "**Edit** `%s`\n```diff\n%s\n```" % (inp.get("file_path", ""), "\n".join(d))
    if name in ("Write", "NotebookEdit"):
        return "**%s** `%s`\n```\n%s\n```" % (name, inp.get("file_path", ""), clip(inp.get("content", "")))
    return "**%s**\n```json\n%s\n```" % (name, clip(json.dumps(inp, indent=1)))


def load(path):
    recs = []
    for line in open(path, encoding="utf-8"):
        try:
            recs.append(json.loads(line))
        except ValueError:
            pass
    return recs


def summarize(recs):
    s = {"title": "", "branch": "", "models": Counter(), "msgs": 0, "files": set(),
         "cost": 0.0, "added": 0, "removed": 0, "start": "", "end": ""}
    for r in recs:
        t = r.get("type")
        if t == "ai-title":
            s["title"] = r.get("aiTitle", "")
        elif t == "cost-state":
            s["cost"] = r.get("totalCostUSD", 0.0)
            s["added"], s["removed"] = r.get("totalLinesAdded", 0), r.get("totalLinesRemoved", 0)
        ts = r.get("timestamp")
        if ts:
            s["start"] = min(s["start"], ts) if s["start"] else ts
            s["end"] = max(s["end"], ts)
        s["branch"] = r.get("gitBranch") or s["branch"]
        if t == "assistant":
            s["msgs"] += 1
            s["models"][r["message"].get("model", "?")] += 1
            for b in blocks(r["message"]):
                if b.get("type") == "tool_use" and b["name"] in ("Edit", "Write", "NotebookEdit"):
                    s["files"].add(b["input"].get("file_path", ""))
    return s


def render_session(recs, s):
    out = ["# %s" % (s["title"] or "(untitled session)"), "",
           "- **Date:** %s → %s (UTC)" % (s["start"], s["end"]),
           "- **Model(s):** %s" % ", ".join("%s x%d" % kv for kv in s["models"].most_common()),
           "- **Branch:** %s" % s["branch"],
           "- **Cost:** $%.2f — lines +%d/-%d" % (s["cost"], s["added"], s["removed"]), ""]
    for r in recs:
        if r.get("type") not in ("user", "assistant") or r.get("isMeta"):
            continue
        who = "User" if r["type"] == "user" else "Claude"
        if r.get("isSidechain"):
            who += " (subagent)"
        parts = []
        for b in blocks(r["message"]):
            k = b.get("type")
            if k == "text":
                txt = b["text"].strip()
                if txt.startswith("<system-reminder>") or txt.startswith("Caveat:"):
                    continue
                parts.append(txt)
            elif k == "thinking" and KEEP_THINKING:
                parts.append("> *thinking:* " + clip(b["thinking"]).replace("\n", "\n> "))
            elif k == "tool_use":
                parts.append(render_tool(b["name"], b.get("input", {})))
            elif k == "tool_result":
                c = b.get("content")
                c = c if isinstance(c, str) else " ".join(x.get("text", "") for x in c or [] if isinstance(x, dict))
                if c.strip():
                    parts.append("<details><summary>result</summary>\n\n```\n%s\n```\n</details>" % clip(c))
        if parts:
            out.append("### %s — %s\n\n%s\n" % (who, r.get("timestamp", ""), "\n\n".join(parts)))
    return "\n".join(out)


os.makedirs(os.path.join(OUT, "sessions"), exist_ok=True)
rows = []
for path in glob.glob(os.path.join(SRC, "*.jsonl")):
    recs = load(path)
    s = summarize(recs)
    if not s["msgs"]:
        continue
    name = "%s-%s.md" % (s["start"][:16].replace(":", "").replace("T", "-"), os.path.basename(path)[:8])
    open(os.path.join(OUT, "sessions", name), "w", encoding="utf-8").write(render_session(recs, s))
    rows.append((s, name))

rows.sort(key=lambda r: r[0]["start"])
idx = ["# LLM usage index — fast-deconv", "",
       "Exported from `~/.claude/projects/`. %d sessions, %s → %s." % (
           len(rows), rows[0][0]["start"][:10], rows[-1][0]["end"][:10]), "",
       "| Date | Session | Models | Msgs | Cost | Lines +/- | Files | Branch |",
       "|---|---|---|---|---|---|---|---|"]
for s, name in rows:
    idx.append("| %s | [%s](sessions/%s) | %s | %d | $%.2f | +%d/-%d | %d | %s |" % (
        s["start"][:16].replace("T", " "), (s["title"] or name)[:48], name,
        ", ".join(s["models"]), s["msgs"], s["cost"], s["added"], s["removed"],
        len(s["files"]), s["branch"]))
tot = lambda k: sum(s[k] for s, _ in rows)
idx.append("| **total** | | | **%d** | **$%.2f** | **+%d/-%d** | | |" % (
    tot("msgs"), tot("cost"), tot("added"), tot("removed")))
open(os.path.join(OUT, "index.md"), "w", encoding="utf-8").write("\n".join(idx) + "\n")
print("wrote %s/index.md and %d session files" % (OUT, len(rows)))
