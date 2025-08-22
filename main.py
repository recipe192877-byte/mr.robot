# ===================== main.py (Merged: M1 + M2 + RuleDigit + SmartPredict + RoundPrior) =====================
# NOTE: This file is the full, ready-to-run main.py. Replace your old main.py with this.
# I preserved your original data loading, CSV_URL, BOT_TOKEN, and other personal config exactly as they were.
# I removed other strategies (S1,S2,S4,S5,M3) and integrated only M1, M2 + the 3 new strategies you provided.
# /train_full added. All missing imports and minor errors fixed.

import telebot, os, re, time, json, threading, requests
import numpy as np
import pandas as pd
from datetime import datetime, date
from flask import Flask
from collections import Counter, defaultdict
from dataclasses import dataclass, field

# ======== CONFIG (kept as-is from your original file) ========
BOT_TOKEN = "8361880658:AAFVMOKWxCIWCy0-X80XCbjB9z00rSunr6c"  # kept untouched (your original)
ADMIN_CHAT_ID = None  # auto-filled from first /start if None
SPREADSHEET_ID = "10wI8T-NzqYsq6L73kPZ_bibuv2dw7xhQAmOr0msvk1A"
CSV_URL = f"https://docs.google.com/spreadsheets/d/{SPREADSHEET_ID}/export?format=csv"

# ======== GLOBALS ========
bot = telebot.TeleBot(BOT_TOKEN)
app = Flask('')

# Data history (kept original names)
digits, rounds_hist, dates_hist = [], [], []

# knobs
TRAIN_RATIO = 0.9

# ======== STORAGE (JSON-safe) ========
learning_storage_file = "learning_data.json"


def _atomic_write_json(path: str, obj: dict):
    import tempfile
    d = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp_path = tempfile.mkstemp(prefix="tmp_", suffix=".json", dir=d)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=2, default=str)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise


# Stats holders (for M1, M2, SmartPredict, RoundPrior)
M1_stats = {
    'name': 'BayesDirichlet',
    'total': 0,
    'ok': 0,
    'acc': 0.0,
    'conf': []
}
M2_stats = {'name': 'HazardGap', 'total': 0, 'ok': 0, 'acc': 0.0, 'conf': []}
SP_stats = {
    'name': 'SmartPredict',
    'total': 0,
    'ok': 0,
    'acc': 0.0,
    'conf': []
}
RP_stats = {'name': 'RoundPrior', 'total': 0, 'ok': 0, 'acc': 0.0, 'conf': []}


def save_learning_data():
    data = {
        "M1": M1_stats,
        "M2": M2_stats,
        "SP": SP_stats,
        "RP": RP_stats,
        "autobet": AUTO.save_state() if 'AUTO' in globals() else {},
        "last_update": datetime.now().isoformat(),
    }
    try:
        _atomic_write_json(learning_storage_file, data)
    except Exception as e:
        print(f"❌ Error saving learning data: {e}")


def load_learning_data():
    try:
        if os.path.exists(learning_storage_file) and os.path.getsize(
                learning_storage_file) > 0:
            try:
                with open(learning_storage_file, "r") as f:
                    data = json.load(f)
            except Exception:
                print("⚠️ learning_data.json corrupt — resetting this run.")
                data = {}
        else:
            data = {}
        for name, store in [("M1", M1_stats), ("M2", M2_stats),
                            ("SP", SP_stats), ("RP", RP_stats)]:
            if isinstance(data.get(name), dict): store.update(data[name])
        if "autobet" in data and "AUTO" in globals():
            AUTO.load_state(data["autobet"])
        print("✅ Learning + AutoBet state loaded")
    except Exception as e:
        print(f"❌ Error loading learning data: {e}")


# ======== UTILS ========
def _norm(v):
    v = np.array(v, dtype=float)
    v[v < 0] = 0.0
    s = v.sum()
    return (v / s) if s > 0 else np.ones(10) / 10.0


def _softmax(x, t=1.0):
    a = np.array(x, dtype=float) / max(t, 1e-9)
    a -= a.max()
    e = np.exp(a)
    s = e.sum()
    return e / s if s > 0 else np.ones_like(a) / len(a)


def _parse_date_flexible(s: str):
    if not s:
        return None
    s = s.strip()
    ISO = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    DMY_S = re.compile(r"^\d{2}/\d{2}/\d{4}$")
    DMY_D = re.compile(r"^\d{2}-\d{2}-\d{4}$")
    MDY_S = re.compile(r"^\d{1,2}/\d{1,2}/\d{4}$")
    try:
        if ISO.match(s):
            dt = pd.to_datetime(s,
                                format="%Y-%m-%d",
                                dayfirst=False,
                                errors="coerce")
        elif DMY_S.match(s):
            dt = pd.to_datetime(s,
                                format="%d/%m/%Y",
                                dayfirst=True,
                                errors="coerce")
        elif DMY_D.match(s):
            dt = pd.to_datetime(s,
                                format="%d-%m-%Y",
                                dayfirst=True,
                                errors="coerce")
        elif MDY_S.match(s):
            dt = pd.to_datetime(s,
                                format="%m/%d/%Y",
                                dayfirst=False,
                                errors="coerce")
        else:
            dt = pd.to_datetime(s, dayfirst=True, errors="coerce")
        return dt.to_pydatetime().date() if pd.notna(dt) else None
    except:
        return None


def load_google_sheets_data():
    global digits, rounds_hist, dates_hist
    old_n = len(digits)
    digits, rounds_hist, dates_hist = [], [], []
    try:
        r = requests.get(CSV_URL, timeout=12)
        r.raise_for_status()
        lines = r.text.strip().split('\n')
        if not lines:
            return False
        headers = [c.strip().strip('"') for c in lines[0].split(',')]
        idx_digit = headers.index('Digit') if 'Digit' in headers else None
        idx_round = headers.index('Round') if 'Round' in headers else None
        idx_date = headers.index('Date') if 'Date' in headers else None
        for ln in lines[1:]:
            if not ln.strip():
                continue
            cols = [c.strip().strip('"') for c in ln.split(',')]
            if idx_digit is None or len(
                    cols) <= idx_digit or not cols[idx_digit].isdigit():
                continue
            val = int(cols[idx_digit])
            if 0 <= val <= 9:
                digits.append(val)
            else:
                continue
            # round
            if idx_round is not None and len(cols) > idx_round:
                try:
                    rounds_hist.append(int(cols[idx_round]))
                except:
                    # keep None if parse fail
                    try:
                        rounds_hist.append(int(float(cols[idx_round])))
                    except:
                        rounds_hist.append(None)
            else:
                rounds_hist.append(None)
            # date
            if idx_date is not None and len(
                    cols) > idx_date and cols[idx_date]:
                dates_hist.append(_parse_date_flexible(cols[idx_date]))
            else:
                dates_hist.append(None)
        print(f"✅ Loaded {len(digits)} rows (Δ {len(digits)-old_n})")
        return True
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False


# ======== M1/M2 (kept from your original main.py) ========
@dataclass
class StrategyBayesDirichlet:
    prior_scale: float = 1.0
    recent_window: int = 30
    sample_count: int = 200
    alpha_global: np.ndarray = field(default_factory=lambda: np.ones(10))

    def train(self, h):
        cnt = Counter(h)
        self.alpha_global = np.array(
            [self.prior_scale + cnt.get(d, 0) for d in range(10)], float)

    def predict(self, h):
        if not h:
            pr = np.ones(10) / 10.0
            return 0, 0.1, pr
        r = h[-self.recent_window:] if len(h) >= self.recent_window else h
        cnt_r = Counter(r)
        alpha = self.alpha_global + np.array(
            [cnt_r.get(d, 0) for d in range(10)], float)
        wins = np.zeros(10)
        for _ in range(self.sample_count):
            s = np.random.dirichlet(alpha)
            wins[np.argmax(s)] += 1
        pr = wins / wins.sum() if wins.sum() > 0 else np.ones(10) / 10.0
        p = int(np.argmax(pr))
        return p, float(pr[p]), pr


@dataclass
class StrategyHazardGap:
    alpha: float = 1.0
    delay_gaps: dict = field(
        default_factory=lambda: {d: []
                                 for d in range(10)})

    def train(self, h):
        self.delay_gaps = {d: [] for d in range(10)}
        last = {d: None for d in range(10)}
        for i, v in enumerate(h):
            if last[v] is not None:
                self.delay_gaps[v].append(i - last[v])
            last[v] = i

    def predict(self, h):
        if not h:
            pr = np.ones(10) / 10.0
            return 0, 0.1, pr
        last = {d: None for d in range(10)}
        for i, v in enumerate(h):
            last[v] = i
        L = len(h)
        scores = np.zeros(10)
        for d in range(10):
            gaps = self.delay_gaps.get(d, [])
            if not gaps:
                scores[d] = 1.0
                continue
            arr = np.array(gaps)
            cur = (L - 1 - last[d]) if last[d] is not None else int(arr.mean())
            cur = max(1, int(cur))
            ge = (arr >= cur).sum()
            eq = (arr == cur).sum()
            scores[d] = (eq + self.alpha * 0.25) / (ge + self.alpha * 2.5)
        pr = _norm(scores)
        p = int(np.argmax(pr))
        return p, float(pr[p]), pr


# instantiate M1, M2
M1 = StrategyBayesDirichlet()
M2 = StrategyHazardGap()

# ======== NEW STRATEGY BLOCKS YOU GAVE (integrated) ========


# 1) RuleDigit (digit_from_patti)
def digit_from_patti(patti: str):
    digits_local = [int(ch) for ch in str(patti) if ch.isdigit()]
    if not digits_local:
        return None
    return sum(digits_local) % 10


# 2) SmartPredict (Order-2 Markov)
class SmartPredict:

    def __init__(self):
        self.transitions = defaultdict(lambda: Counter())
        self.trained = False

    def reset(self):
        self.transitions.clear()
        self.trained = False

    def train(self, seq):
        self.reset()
        seq2 = [int(x) for x in seq if x is not None]
        for i in range(2, len(seq2)):
            prev2 = (seq2[i - 2], seq2[i - 1])
            self.transitions[prev2][seq2[i]] += 1
        self.trained = len(self.transitions) > 0

    def predict(self, prev2, topn=10):
        if not self.trained:
            # return uniform probabilities
            return [(d, 1.0 / 10.0) for d in range(10)]
        counter = self.transitions.get(prev2)
        if not counter:
            return [(d, 1.0 / 10.0) for d in range(10)]
        total = sum(counter.values()) + 10  # Laplace smoothing
        probs = {d: (counter.get(d, 0) + 1) / total for d in range(10)}
        ranked = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:topn]
        return ranked


# 3) RoundPrior (time-decayed by round)
ROUNDS = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8']
R2I = {r: i for i, r in enumerate(ROUNDS)}


class RoundPrior:

    def __init__(self, decay=0.997):
        self.decay = decay
        self.by_round = defaultdict(lambda: Counter())
        self.trained = False

    def reset(self):
        self.by_round.clear()
        self.trained = False

    def train(self, rounds, digits_list, dates=None):
        self.reset()
        recs = []
        for r, g, d in zip(rounds, digits_list, dates or [''] * len(rounds)):
            try:
                g = int(g)
                if isinstance(r, int):
                    # original rounds_hist often had integers, map to 'R{n}'
                    rr = f"R{r}"
                else:
                    rr = r
                if rr not in ROUNDS:
                    continue
                dt = None
                if d:
                    try:
                        dt = datetime.fromisoformat(str(d))
                    except Exception:
                        pass
                recs.append((dt.date() if dt else None, rr, g))
            except Exception:
                continue
        recs.sort(key=lambda x:
                  (x[0] or datetime(1970, 1, 1).date(), R2I.get(x[1], 0)))
        last_day = None
        for day, rr, g in recs:
            if day is not None and last_day is not None and day != last_day:
                # apply decay daily
                for rkey in list(self.by_round.keys()):
                    for k in list(self.by_round[rkey].keys()):
                        self.by_round[rkey][k] *= self.decay
            last_day = day if day is not None else last_day
            self.by_round[rr][g] += 1.0
        self.trained = True

    def dist(self, r):
        if isinstance(r, int):
            r = f"R{r}"
        c = self.by_round.get(r, None)
        if not c:
            return [1.0 / 10.0] * 10
        tot = sum(c.values()) + 10
        return [(c.get(d, 0.0) + 1) / tot for d in range(10)]


# instantiate user strategies
SP = SmartPredict()
RP = RoundPrior()


# ======== ENSEMBLE: combine M1,M2,SmartPredict,RoundPrior ========
def ensemble_predict(history):
    """
    Combine M1, M2, SP, RP into a single probability mixture.
    Returns: pred, conf, top3_list, mix_vector
    """
    # probabilities vectors for each model
    pr_m1 = np.ones(10) / 10.0
    pr_m2 = np.ones(10) / 10.0
    pr_sp = np.ones(10) / 10.0
    pr_rp = np.ones(10) / 10.0

    try:
        _, _, pm1 = M1.predict(history)
        pr_m1 = np.array(pm1, dtype=float)
    except Exception:
        pass
    try:
        _, _, pm2 = M2.predict(history)
        pr_m2 = np.array(pm2, dtype=float)
    except Exception:
        pass
    try:
        prev2 = (history[-2], history[-1]) if len(history) >= 2 else None
        sp_scores = SP.predict(prev2, topn=10)
        # convert sp_scores list of (d,score) to vector
        pr_sp = np.ones(10) * 1e-12
        for d, s in sp_scores:
            pr_sp[d] = float(s)
        pr_sp = pr_sp / pr_sp.sum()
    except Exception:
        pass
    try:
        # use last known round if available
        last_round = None
        if rounds_hist:
            last_round_raw = rounds_hist[-1]
            # next round guess (for RP dist we might want next round, but we'll use next)
            try:
                idx = int(last_round_raw)
                next_r = f"R{(idx % 8) + 1}"
            except Exception:
                # if stored as 'R1' etc
                try:
                    next_r = f"R{int(str(last_round_raw).strip().lstrip('R')) + 1}"
                except Exception:
                    next_r = None
            if next_r:
                pr_rp = np.array(RP.dist(next_r), dtype=float)
    except Exception:
        pass

    # weights (tunable)
    w = {"M1": 0.22, "M2": 0.18, "SP": 0.40, "RP": 0.20}
    mix = _norm(w["M1"] * pr_m1 + w["M2"] * pr_m2 + w["SP"] * pr_sp +
                w["RP"] * pr_rp)
    pred = int(np.argmax(mix))
    conf = float(mix[pred])
    top3 = np.argsort(-mix)[:3].tolist()
    return pred, conf, top3, mix


# ======== TRAIN/EVAL (leak-proof) ========
def _bump(stats, pred, actual, conf):
    stats['total'] += 1
    if pred == actual:
        stats['ok'] += 1
    stats['acc'] = (stats['ok'] / stats['total'] *
                    100.0) if stats['total'] else 0.0
    stats['conf'].append(float(conf))
    if len(stats['conf']) > 200:
        stats['conf'] = stats['conf'][-200:]


def calculate_accuracy_leakproof():
    """Leak-proof accuracy calculation for M1, M2, SP, RP using TRAIN_RATIO split."""
    if len(digits) < 80:
        return
    split = int(len(digits) * TRAIN_RATIO)
    train_d, test_d = digits[:split], digits[split:]

    # reset stats
    for s in (M1_stats, M2_stats, SP_stats, RP_stats):
        s.update({'total': 0, 'ok': 0, 'acc': 0.0, 'conf': []})

    # train on train_d
    M1.train(train_d)
    M2.train(train_d)
    SP.train(train_d)
    RP.train(rounds_hist[:split], train_d,
             dates_hist[:split] if dates_hist else None)

    # evaluate on test_d sequentially (simulate streaming)
    for i in range(len(test_d)):
        hist = train_d + test_d[:i]
        actual = test_d[i]
        try:
            p, c, _ = M1.predict(hist)
            _bump(M1_stats, p, actual, c)
        except Exception:
            pass
        try:
            p, c, _ = M2.predict(hist)
            _bump(M2_stats, p, actual, c)
        except Exception:
            pass
        try:
            prev2 = (hist[-2], hist[-1]) if len(hist) >= 2 else None
            preds = SP.predict(prev2, topn=3)
            p_sp = preds[0][0] if preds else 0
            _bump(SP_stats, p_sp, actual, preds[0][1] if preds else 0.0)
        except Exception:
            pass
        try:
            # RP prediction: use the same round logic as training slice
            r = rounds_hist[split + i] if len(rounds_hist) > (split +
                                                              i) else None
            if r is None:
                # fallback to previous round
                r = rounds_hist[-1] if rounds_hist else None
            prd = RP.dist(r) if RP.trained else [1.0 / 10.0] * 10
            p_rp = int(np.argmax(prd))
            _bump(RP_stats, p_rp, actual, float(prd[p_rp]))
        except Exception:
            pass


def train_all_full():
    """Train M1,M2,SmartPredict,RoundPrior on full digits history."""
    try:
        M1.train(digits)
        M2.train(digits)
        SP.train(digits)
        RP.train(rounds_hist, digits, dates_hist)
        save_learning_data()
        return True
    except Exception as e:
        print("🔴 train_all_full error:", e)
        return False


# ======== AUTO BET PLANNER (kept mostly same, but uses new ensemble_predict) ========
class AutoBetPlanner:

    def __init__(self):
        # defaults as requested
        self.capital = 3800
        self.bet_per_digit = 10
        self.daily_risk_frac = 0.10  # 10% capital as daily risk cap
        self.daily_target = 400  # ~₹400 profit
        self.daily_stop_loss = None  # if None, computed from risk_frac
        self.risk_threshold = 0.20  # if round risk > 20% of daily risk OR conf<40% => SKIP
        self.enabled = False
        self.chat_id = None
        self._reset_day_state()

    def _reset_day_state(self):
        self.day = date.today()
        self.day_bet_total = 0
        self.day_win_total = 0
        self.day_net = 0
        self.round_log = []
        self.last_seen_n = 0

    def _ensure_today(self):
        if self.day != date.today():
            self._reset_day_state()

    def daily_risk_budget(self):
        if self.daily_stop_loss is not None:
            return max(0, float(self.daily_stop_loss))
        return max(50.0, float(self.capital) * float(self.daily_risk_frac))

    def save_state(self):
        return {
            "capital": self.capital,
            "bet_per_digit": self.bet_per_digit,
            "daily_risk_frac": self.daily_risk_frac,
            "daily_target": self.daily_target,
            "daily_stop_loss": self.daily_stop_loss,
            "risk_threshold": self.risk_threshold,
            "enabled": self.enabled,
        }

    def load_state(self, d):
        try:
            self.capital = d.get("capital", self.capital)
            self.bet_per_digit = d.get("bet_per_digit", self.bet_per_digit)
            self.daily_risk_frac = d.get("daily_risk_frac",
                                         self.daily_risk_frac)
            self.daily_target = d.get("daily_target", self.daily_target)
            self.daily_stop_loss = d.get("daily_stop_loss",
                                         self.daily_stop_loss)
            self.risk_threshold = d.get("risk_threshold", self.risk_threshold)
            self.enabled = d.get("enabled", self.enabled)
        except Exception as e:
            print("AutoBet load_state err:", e)

    def _consensus_mix(self, pr_list):
        mix = _norm(sum(pr_list))
        top3 = np.argsort(-mix)[:3].tolist()
        return top3, float(mix[top3[0]]), mix

    def select_digits(self):
        # use ensemble + strategy consensus to choose up to 3 digits
        pred, conf, top3e, mix = ensemble_predict(digits)
        # consensus from top1 of each model
        votes = Counter()
        try:
            p1, _, _ = M1.predict(digits)
            votes[p1] += 1
        except:
            pass
        try:
            p2, _, _ = M2.predict(digits)
            votes[p2] += 1
        except:
            pass
        try:
            prev2 = (digits[-2], digits[-1]) if len(digits) >= 2 else None
            sp_scores = SP.predict(prev2, topn=1)
            if sp_scores:
                votes[sp_scores[0][0]] += 1
        except:
            pass
        # base picks from ensemble top3
        picks = []
        for d in top3e:
            if votes[d] >= 2 or mix[d] >= 0.18:
                picks.append(d)
        if len(picks) == 0:
            picks = [top3e[0]]
        if len(picks) > 3:
            picks = sorted(picks, key=lambda x: -mix[x])[:3]
        conf_avg = float(sum(mix[d] for d in picks) / len(picks))
        return picks, conf_avg, mix

    def risk_score(self, num_digits, conf):
        cost = num_digits * self.bet_per_digit
        risk_budget = self.daily_risk_budget()
        frac = cost / max(1.0, risk_budget)
        conf_penalty = max(0.0, 0.4 - conf) / 0.4
        return frac + 0.5 * conf_penalty, cost, risk_budget

    def recommend_text(self, picks, conf, risk_sc, cost):
        risk_tag = "Low ✅" if risk_sc < self.risk_threshold else "High ⚠️"
        rec = "PLAY" if risk_sc < self.risk_threshold and conf >= 0.40 else "SKIP ❌"
        return risk_tag, rec

    def handle_new_data(self):
        self._ensure_today()
        if not self.enabled or self.chat_id is None:
            return
        n = len(digits)
        if n <= self.last_seen_n:
            return
        self.last_seen_n = n
        if self.day_net >= self.daily_target:
            bot.send_message(
                self.chat_id,
                f"🛑 *Target Reached* — Net +₹{self.day_net}.\nAutoBet paused for today.",
                parse_mode='HTML')
            self.enabled = False
            save_learning_data()
            return
        if -self.day_net >= self.daily_risk_budget():
            bot.send_message(
                self.chat_id,
                f"🛑 *Stop-Loss Hit* — Net -₹{abs(self.day_net)}.\nAutoBet paused for today.",
                parse_mode='HTML')
            self.enabled = False
            save_learning_data()
            return
        picks, conf, mix = self.select_digits()
        risk_sc, cost, budget = self.risk_score(len(picks), conf)
        risk_tag, rec = self.recommend_text(picks, conf, risk_sc, cost)
        msg = (
            f"📌 *Next Round Suggestion*\n"
            f"Digits: *{', '.join(map(str,picks))}*\n"
            f"Bet: ₹{self.bet_per_digit} each → *₹{cost}*\n"
            f"Risk: {risk_tag}  |  Confidence: *{conf*100:.1f}%*\n\n"
            f"👉 Recommended: {rec}\n"
            f"— Budget today: ₹{budget:.0f} | Used: ₹{self.day_bet_total}\n")
        bot.send_message(self.chat_id, msg, parse_mode='HTML')

    def record_result(self, played_digits, actual_digit):
        if not played_digits:
            return
        cost = len(played_digits) * self.bet_per_digit
        win = 0
        if actual_digit in played_digits:
            win = 90
            win = int((self.bet_per_digit / 10.0) * 90)
        self.day_bet_total += cost
        self.day_win_total += win
        self.day_net = self.day_win_total - self.day_bet_total
        self.round_log.append({
            "played": played_digits,
            "actual": actual_digit,
            "cost": cost,
            "win": win
        })


AUTO = AutoBetPlanner()


# ======== MINI WEB STATUS (keep_alive) ========
@app.route("/healthz")
def _healthz():
    return "ok"


@app.route('/')
def home():
    try:
        return f"""
        <h1>🤖 Kolkata FF — Auto Bet Planner (Merged)</h1>
        <p>N={len(digits)} | Capital: ₹{AUTO.capital} | Bet/Digit: ₹{AUTO.bet_per_digit}</p>
        <p>🎯 Target: ₹{AUTO.daily_target} | Stop-loss: ₹{AUTO.daily_risk_budget():.0f}</p>
        <p>🟢 AutoBet: {"ON" if AUTO.enabled else "OFF"} | Risk-th: {AUTO.risk_threshold:.2f}</p>
        <p>📈 Acc: M1 {M1_stats['acc']:.1f}% | M2 {M2_stats['acc']:.1f}% | SP {SP_stats['acc']:.1f}% | RP {RP_stats['acc']:.1f}%</p>
        """
    except Exception:
        return "ok"


# ======== TELEGRAM UI COMMANDS ========
WELCOME = """🤖 <b>Kolkata FF Auto Bet Planner Ready!</b>

Use:
• /autobet on — start auto suggestions
• /autobet off — stop
• /betsummary — see today summary
• /set_capital 3800
• /set_bet 10
• /set_risk 0.2
• /refresh — reload sheet & re-evaluate
• /accuracy — current accuracies
• /train_full — retrain M1+M2+SmartPredict+RoundPrior on full data
• /patti 467 — convert patti to digit
• /predict [R1..R8] — get top-3 forecast
"""


@bot.message_handler(commands=['start'])
def start_cmd(m):
    global ADMIN_CHAT_ID
    if ADMIN_CHAT_ID is None:
        try:
            ADMIN_CHAT_ID = m.chat.id
        except Exception:
            pass
    AUTO.chat_id = m.chat.id
    msg = WELCOME
    bot.reply_to(m, msg, parse_mode='HTML')


@bot.message_handler(commands=['predict'])
def predict_cmd(m):
    if len(digits) < 5:
        bot.reply_to(m, "❌ Not enough data.")
        return
    try:
        parts = m.text.strip().split()
        if len(parts) > 1 and parts[1].upper() in ROUNDS:
            rn = parts[1].upper()
        else:
            # next round guess
            if rounds_hist and any(r is not None for r in rounds_hist):
                try:
                    _valid_rounds = [r for r in rounds_hist if r is not None]
                    _next_round_val = (max([
                        int(r) for r in _valid_rounds if
                        isinstance(r, (int, float, str)) and str(r).isdigit()
                    ]) + 1) if _valid_rounds else None
                    rn = f"R{_next_round_val}" if _next_round_val else 'R1'
                except Exception:
                    rn = 'R1'
            else:
                rn = 'R1'
        pred, conf, top3, mix = ensemble_predict(digits)
        top3_display = top3
        s = f"📢 *Round {rn} Prediction*\n\n🎯 Ensemble → *{pred}* | Conf: {conf*100:.1f}%\nTop-3: {top3_display}\nN={len(digits)}\n⏰ {datetime.now().strftime('%H:%M:%S')}"
        bot.reply_to(m, s, parse_mode='HTML')
    except Exception as e:
        bot.reply_to(m, f"❌ Predict error: {e}")


@bot.message_handler(commands=['patti'])
def patti_cmd(m):
    parts = m.text.strip().split()
    if len(parts) < 2:
        return bot.reply_to(m, "Usage: /patti 467")
    val = digit_from_patti(parts[1])
    bot.reply_to(m,
                 f"Patti {parts[1]} → Digit {val if val is not None else '❌'}",
                 parse_mode='HTML')


@bot.message_handler(commands=['refresh'])
def refresh_cmd(m):
    ok = load_google_sheets_data()
    if ok:
        SP.train(digits)
        RP.train(rounds_hist, digits, dates_hist)
        calculate_accuracy_leakproof()
        save_learning_data()
        bot.reply_to(
            m,
            f"✅ Data refreshed. N={len(digits)}\nM1:{M1_stats['acc']:.2f}% M2:{M2_stats['acc']:.2f}% SP:{SP_stats['acc']:.2f}% RP:{RP_stats['acc']:.2f}%",
            parse_mode='HTML')
        AUTO.handle_new_data()
    else:
        bot.reply_to(m, "❌ Failed to refresh data.")


@bot.message_handler(commands=['train'])
def train_cmd(m):
    try:
        load_google_sheets_data()
        SP.train(digits)
        RP.train(rounds_hist, digits, dates_hist)
        save_learning_data()
        bot.reply_to(m,
                     f"✅ Models trained on available data. N={len(digits)}",
                     parse_mode='HTML')
    except Exception as e:
        bot.reply_to(m, f"❌ Train error: {e}")


@bot.message_handler(commands=['train_full'])
def train_full_cmd(m):
    try:
        load_google_sheets_data()
        ok = train_all_full()
        if ok:
            calculate_accuracy_leakproof()
            bot.reply_to(
                m,
                f"✅ Full retrain done. N={len(digits)}\nM1:{M1_stats['acc']:.2f}% M2:{M2_stats['acc']:.2f}% SP:{SP_stats['acc']:.2f}% RP:{RP_stats['acc']:.2f}%",
                parse_mode='HTML')
        else:
            bot.reply_to(m,
                         "❌ train_full failed. Check logs.",
                         parse_mode='HTML')
    except Exception as e:
        bot.reply_to(m, f"❌ Train_full error: {e}")


@bot.message_handler(commands=['accuracy'])
def acc_cmd(m):
    msg = f"""📊 *Accuracies (Leak-Proof 90/10)*
M1 Bayes: {M1_stats['acc']:.2f}%
M2 Hazard: {M2_stats['acc']:.2f}%
SmartPredict: {SP_stats['acc']:.2f}%
RoundPrior: {RP_stats['acc']:.2f}%
"""
    bot.reply_to(m, msg, parse_mode='HTML')


@bot.message_handler(commands=['autobet'])
def autobet_cmd(m):
    parts = m.text.strip().split()
    if len(parts) < 2 or parts[1] not in ("on", "off"):
        bot.reply_to(m, "Usage: /autobet on | off")
        return
    AUTO.chat_id = m.chat.id
    if parts[1] == "on":
        AUTO.enabled = True
        AUTO._ensure_today()
        bot.reply_to(
            m,
            f"🟢 AutoBet *ON* — Target ₹{AUTO.daily_target}, Stop-loss ~₹{int(AUTO.daily_risk_budget())}\n"
            f"Bet/Digit ₹{AUTO.bet_per_digit}, Risk-th {AUTO.risk_threshold:.2f}",
            parse_mode='HTML')
        AUTO.handle_new_data()
    else:
        AUTO.enabled = False
        bot.reply_to(m, "🔴 AutoBet *OFF*", parse_mode='HTML')
    save_learning_data()


@bot.message_handler(commands=['betsummary'])
def betsummary_cmd(m):
    AUTO._ensure_today()
    msg = (
        f"📊 *Daily Summary*\n"
        f"Total Bets: ₹{AUTO.day_bet_total}\n"
        f"Wins: ₹{AUTO.day_win_total}\n"
        f"Net: {'+' if AUTO.day_net>=0 else ''}₹{AUTO.day_net}\n"
        f"Target Hit: {'Yes' if AUTO.day_net>=AUTO.daily_target else 'No'}\n"
        f"Stop-loss Triggered: {'Yes' if -AUTO.day_net>=AUTO.daily_risk_budget() else 'No'}")
    bot.reply_to(m, msg, parse_mode='HTML')


@bot.message_handler(commands=['set_capital'])
def set_capital_cmd(m):
    try:
        val = float(m.text.strip().split()[1])
        AUTO.capital = max(100.0, val)
        bot.reply_to(
            m,
            f"✅ Capital set to ₹{int(AUTO.capital)}\nStop-loss budget ~₹{int(AUTO.daily_risk_budget())}"
        )
        save_learning_data()
    except Exception:
        bot.reply_to(m, "Usage: /set_capital 3800")


@bot.message_handler(commands=['set_bet'])
def set_bet_cmd(m):
    try:
        val = float(m.text.strip().split()[1])
        AUTO.bet_per_digit = max(1.0, val)
        bot.reply_to(m, f"✅ Bet per digit set to ₹{int(AUTO.bet_per_digit)}")
        save_learning_data()
    except Exception:
        bot.reply_to(m, "Usage: /set_bet 10")


@bot.message_handler(commands=['set_risk'])
def set_risk_cmd(m):
    try:
        val = float(m.text.strip().split()[1])
        AUTO.risk_threshold = min(0.8, max(0.05, val))
        bot.reply_to(
            m,
            f"✅ Risk-threshold set to {AUTO.risk_threshold:.2f}\n(lower = stricter SKIP; higher = more PLAY)"
        )
        save_learning_data()
    except Exception:
        bot.reply_to(m, "Usage: /set_risk 0.2")


# ======== BACKGROUND TASKS (auto_refresh, keep_alive, self_ping) ========
def auto_refresh():
    while True:
        time.sleep(300)  # 5 min
        old = len(digits)
        if load_google_sheets_data():
            if len(digits) > old:
                # retrain light modules
                try:
                    SP.train(digits)
                    RP.train(rounds_hist, digits, dates_hist)
                    calculate_accuracy_leakproof()
                    save_learning_data()
                    print(
                        f"[Auto] New data: {len(digits)} (+{len(digits)-old})")
                    AUTO.handle_new_data()
                except Exception as e:
                    print("[Auto] retrain error:", e)
            else:
                print(f"[Auto] Checked: {len(digits)}")


def keep_alive():

    def run():
        try:
            app.run(host="0.0.0.0", port=3000)
        except Exception as e:
            print("Keep-alive server error:", e)

    t = threading.Thread(target=run)
    t.daemon = True
    t.start()


def self_ping():
    # Multiple fallback options for Replit URL
    possible_urls = []

    # Try REPLIT_DEV_DOMAIN first (new Replit format)
    replit_dev_domain = os.getenv('REPLIT_DEV_DOMAIN')
    if replit_dev_domain:
        possible_urls.append(f"https://{replit_dev_domain}")

    # Try constructing from REPL_SLUG and REPL_OWNER
    repl_slug = os.getenv('REPL_SLUG')
    repl_owner = os.getenv('REPL_OWNER')
    if repl_slug and repl_owner and repl_slug != 'workspace':
        possible_urls.append(f"https://{repl_slug}.{repl_owner}.replit.dev")
        possible_urls.append(f"https://{repl_slug}.{repl_owner}.repl.co")

    # Try getting Replit URL from other environment variables
    replit_url = os.getenv('REPLIT_URL')
    if replit_url:
        possible_urls.append(replit_url)

    # Fallback to localhost
    possible_urls.append("http://localhost:3000")
    possible_urls.append("http://0.0.0.0:3000")

    while True:
        success = False
        for url in possible_urls:
            try:
                response = requests.get(url + "/healthz", timeout=5)
                if response.status_code == 200:
                    print(f"🔄 Self-ping successful: {url}/healthz")
                    success = True
                    break
            except Exception:
                continue

        if not success:
            print("⚠️ All self-ping URLs failed, using localhost fallback")

        time.sleep(300)


def initialize():
    print("🚀 Starting Merged Bot (M1+M2+RuleDigit+SmartPredict+RoundPrior)")
    load_learning_data()
    load_google_sheets_data()
    SP.train(digits)
    RP.train(rounds_hist, digits, dates_hist)
    calculate_accuracy_leakproof()
    keep_alive()
    threading.Thread(target=self_ping, daemon=True).start()
    threading.Thread(target=auto_refresh, daemon=True).start()


def run_bot():
    while True:
        try:
            print("🤖 Bot polling started...")
            bot.polling(none_stop=True, timeout=10)
        except Exception as e:
            print(f"❌ Bot crashed: {e}")
            print("🔄 Restarting in 5 seconds...")
            time.sleep(5)


if __name__ == "__main__":
    initialize()
    run_bot()
