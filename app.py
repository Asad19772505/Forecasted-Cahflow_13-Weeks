# app.py
import io
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Interactive Cash Forecast & Financial Management Dashboard",
                   layout="wide", page_icon="💵")

# -----------------------------
# Helpers & Defaults
# -----------------------------
TODAY = datetime.today().date()

def start_of_week(d):
    d = pd.to_datetime(d).date()
    return d - timedelta(days=d.weekday())  # Monday

def week_index(d, base_start):
    return int((start_of_week(d) - base_start).days // 7)

def empty_13w():
    cols = [f"Week {i+1}" for i in range(13)]
    out = pd.DataFrame({
        "Category":[
            "CASH INFLOWS", "Collections", "Other Inflows", "Total Inflows",
            "CASH OUTFLOWS", "Operating Expenses", "Payroll", "AP Payments", "Capex", "Debt Service",
            "Total Outflows", "Net Cash Flow", "Beginning Cash", "Ending Cash"
        ]
    })
    for c in cols:
        out[c] = 0.0
    return out

def money(x):
    try:
        return f"${float(x):,.2f}"
    except Exception:
        return "$0.00"

def prep_upload_area():
    col1, col2, col3, col4 = st.columns([1.2,1.2,1.2,1.0])
    with col1:
        st.caption(" ")
        ap_file = st.file_uploader("Import A/P Aging", type=["csv","xlsx"], key="ap")
    with col2:
        st.caption(" ")
        ar_file = st.file_uploader("Import A/R Aging", type=["csv","xlsx"], key="ar")
    with col3:
        st.caption(" ")
        gl_file = st.file_uploader("Import GL (90 Days)", type=["csv","xlsx"], key="gl")
    with col4:
        st.caption(" ")
        run = st.button("Generate Cash Flow", use_container_width=True, type="primary")
    return ap_file, ar_file, gl_file, run

def read_any(file):
    if file is None:
        return None
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)

def coerce_date(x):
    try:
        return pd.to_datetime(x).date()
    except Exception:
        return None

def ensure_columns(df, required):
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.warning(f"Missing columns: {missing}")
        return False
    return True

def template_bytes(kind="ap"):
    """Create downloadable templates."""
    if kind == "ap":
        df = pd.DataFrame({
            "Vendor":["Acme Ltd","Global Supplies"],
            "InvoiceNumber":["A-1001","G-3390"],
            "InvoiceDate":[TODAY - timedelta(days=20), TODAY - timedelta(days=35)],
            "DueDate":[TODAY + timedelta(days=10), TODAY + timedelta(days=25)],
            "Amount":[12500, 34800],
            "Currency":["USD","USD"]
        })

    elif kind == "ar":
        df = pd.DataFrame({
            "Customer":["Client A","Client B"],
            "InvoiceNumber":["INV-501","INV-502"],
            "InvoiceDate":[TODAY - timedelta(days=18), TODAY - timedelta(days=40)],
            "DueDate":[TODAY + timedelta(days=14), TODAY + timedelta(days=30)],
            "Amount":[22000, 18500],
            "Currency":["USD","USD"]
        })

    else:  # gl (90 rows exactly)
        n = 90
        dates = [TODAY - timedelta(days=i) for i in range(n)][::-1]  # <-- FIXED: 90 items
        # Make accounts exactly 90 rows
        pattern = ["Cash", "Expense", "Expense", "Revenue"]
        accounts = (pattern * (n // len(pattern))) + pattern[:(n % len(pattern))]
        desc = ["Opening/Activity"] * n
        debit = [0] * n
        credit = [0] * n
        # Synthetic amounts (positive revenue, negative expense)
        amt = []
        for a in accounts:
            if a.lower() == "revenue":
                amt.append(int(np.random.randint(8000, 15000)))
            elif a.lower() == "expense":
                amt.append(int(-np.random.randint(4000, 9000)))
            else:
                amt.append(0)
        df = pd.DataFrame({
            "Date": dates,
            "Account": accounts,
            "Description": desc,
            "Debit": debit,
            "Credit": credit,
            "Amount": amt
        })

    buf = io.BytesIO()
    df.to_excel(buf, index=False, sheet_name="Template")
    buf.seek(0)
    return buf

def kpi_card(label, value):
    st.markdown(
        f"""
        <div style="padding:16px;border:1px solid #e9eef4;border-radius:12px;background:#fff;">
            <div style="color:#64748b;font-size:12px;text-transform:uppercase;letter-spacing:.06em;">{label}</div>
            <div style="font-size:28px;font-weight:700;margin-top:4px;">{value}</div>
        </div>
        """, unsafe_allow_html=True
    )

# -----------------------------
# Header
# -----------------------------
st.markdown(
    """
    <div style="padding:24px 16px;background:linear-gradient(180deg,#0b2740,#0f3657);color:#dbeafe;border-radius:12px;margin-bottom:10px;">
        <div style="font-size:36px;font-weight:700;text-align:center;letter-spacing:.02em;">[NEW CLIENT NAME]</div>
        <div style="text-align:center;opacity:.9;font-size:16px;margin-top:6px;">
            Interactive Cash Forecast & Financial Management Dashboard
        </div>
    </div>
    """, unsafe_allow_html=True
)

# -----------------------------
# Uploads row
# -----------------------------
ap_file, ar_file, gl_file, run_fc = prep_upload_area()
st.divider()
tabs = st.tabs(["Payables Management", "Receivables", "Cash Forecast"])

# -----------------------------
# Read inputs
# -----------------------------
ap_df = read_any(ap_file) if ap_file else None
ar_df = read_any(ar_file) if ar_file else None
gl_df = read_any(gl_file) if gl_file else None

# -----------------------------
# Payables Tab
# -----------------------------
with tabs[0]:
    st.subheader("A/P Aging")
    st.caption("Expected columns: Vendor, InvoiceNumber, InvoiceDate, DueDate, Amount, Currency")
    if ap_df is not None:
        st.dataframe(ap_df, use_container_width=True, height=320)
    else:
        st.info("Upload an A/P file to view and include in the forecast.")
    st.download_button("Download A/P Template (.xlsx)", data=template_bytes("ap"),
                       file_name="ap_template.xlsx", use_container_width=True)

# -----------------------------
# Receivables Tab
# -----------------------------
with tabs[1]:
    st.subheader("A/R Aging")
    st.caption("Expected columns: Customer, InvoiceNumber, InvoiceDate, DueDate, Amount, Currency")
    if ar_df is not None:
        st.dataframe(ar_df, use_container_width=True, height=320)
    else:
        st.info("Upload an A/R file to view and include in the forecast.")
    st.download_button("Download A/R Template (.xlsx)", data=template_bytes("ar"),
                       file_name="ar_template.xlsx", use_container_width=True)

# -----------------------------
# Cash Forecast Tab
# -----------------------------
with tabs[2]:
    st.subheader("Cash Forecast")
    st.caption("Dynamic Forecast: This forecast automatically updates based on your payables/receivables selections. "
               "Selected payments will impact cash flow in their scheduled payment weeks.")
    
    left, mid, right = st.columns([1.2,1.2,1.2])
    with left:
        base_cash = st.number_input("Current Cash (Beginning of Week 1)", min_value=0.0, value=0.0, step=1000.0, format="%.2f")
    with mid:
        base_start = start_of_week(TODAY)
        st.text_input("Forecast Start Week (Monday)", value=str(base_start), disabled=True)
    with right:
        include_gl_ops = st.checkbox("Use last 90 days GL to estimate weekly Opex/Payroll", value=True)

    table = empty_13w()
    week_cols = [c for c in table.columns if c.startswith("Week")]

    # Inflows from AR
    if ar_df is not None and ensure_columns(ar_df, ["DueDate","Amount"]):
        df = ar_df.copy()
        df["DueDate"] = df["DueDate"].apply(coerce_date)
        df = df.dropna(subset=["DueDate"])
        df["w"] = df["DueDate"].apply(lambda d: week_index(d, base_start))
        for i in range(13):
            table.loc[table["Category"].eq("Collections"), f"Week {i+1}"] = float(df.loc[df["w"].eq(i), "Amount"].sum())

    # Outflows from AP
    if ap_df is not None and ensure_columns(ap_df, ["DueDate","Amount"]):
        df = ap_df.copy()
        df["DueDate"] = df["DueDate"].apply(coerce_date)
        df = df.dropna(subset=["DueDate"])
        df["w"] = df["DueDate"].apply(lambda d: week_index(d, base_start))
        for i in range(13):
            table.loc[table["Category"].eq("AP Payments"), f"Week {i+1}"] = float(df.loc[df["w"].eq(i), "Amount"].sum())

    # Estimate Opex/Payroll from GL (simple heuristic)
    weekly_opex = 0.0
    weekly_payroll = 0.0
    if include_gl_ops and gl_df is not None:
        g = gl_df.copy()
        # Guess column names
        guessed = {"date":"Date","account":"Account","amount":"Amount"}
        for c in list(g.columns):
            lc = c.lower()
            if "date" in lc: guessed["date"] = c
            if "acc" in lc:  guessed["account"] = c
            if "amount" in lc or "debit" in lc or "credit" in lc: guessed["amount"] = c

        g[guessed["date"]] = pd.to_datetime(g[guessed["date"]], errors="coerce")
        g = g.dropna(subset=[guessed["date"]])
        g["Amount"] = pd.to_numeric(g[guessed["amount"]], errors="coerce").fillna(0.0)

        acct = g[guessed["account"]].astype(str).str.lower()
        opex = g.loc[acct.str.contains("expens|utilities|rent|marketing|admin"), "Amount"]
        payroll = g.loc[acct.str.contains("payroll|salary|wage"), "Amount"]

        if len(opex):
            weekly_opex = abs(opex.mean()) * 5  # rough weekly scale
        if len(payroll):
            weekly_payroll = abs(payroll.mean()) * 5

    for i in range(13):
        wcol = f"Week {i+1}"
        inflows = table.loc[table["Category"].eq("Collections"), wcol] + table.loc[table["Category"].eq("Other Inflows"), wcol]
        table.loc[table["Category"].eq("Total Inflows"), wcol] = inflows

        if weekly_opex > 0:
            table.loc[table["Category"].eq("Operating Expenses"), wcol] = weekly_opex
        if weekly_payroll > 0:
            table.loc[table["Category"].eq("Payroll"), wcol] = weekly_payroll

        outflows = (table.loc[table["Category"].eq("Operating Expenses"), wcol] +
                    table.loc[table["Category"].eq("Payroll"), wcol] +
                    table.loc[table["Category"].eq("AP Payments"), wcol] +
                    table.loc[table["Category"].eq("Capex"), wcol] +
                    table.loc[table["Category"].eq("Debt Service"), wcol])

        table.loc[table["Category"].eq("Total Outflows"), wcol] = outflows
        table.loc[table["Category"].eq("Net Cash Flow"), wcol] = inflows - outflows

    # Roll-forward
    table.loc[table["Category"].eq("Beginning Cash"), "Week 1"] = base_cash
    for i in range(13):
        w = f"Week {i+1}"
        beg = float(table.loc[table["Category"].eq("Beginning Cash"), w])
        net = float(table.loc[table["Category"].eq("Net Cash Flow"), w])
        end = beg + net
        table.loc[table["Category"].eq("Ending Cash"), w] = end
        if i < 12:
            wnext = f"Week {i+2}"
            table.loc[table["Category"].eq("Beginning Cash"), wnext] = end

    # KPIs
    end_w1 = float(table.loc[table["Category"].eq("Ending Cash"), "Week 1"])
    ending_13 = float(table.loc[table["Category"].eq("Ending Cash"), "Week 13"])
    end_series = table.loc[table["Category"].eq("Ending Cash"), week_cols].values.flatten()  # <-- simplified
    min_idx = int(np.argmin(end_series)) + 1  # <-- display as Week 1..13
    min_val = float(np.min(end_series))

    k1, k2, k3, k4 = st.columns(4)
    with k1: kpi_card("Current Cash", money(base_cash))
    with k2: kpi_card("Week 1 Ending Cash", money(end_w1))
    with k3: kpi_card("Lowest Cash Week", f"Week {min_idx}: {money(min_val)}")
    with k4: kpi_card("13-Week Ending Cash", money(ending_13))

    st.info("Dynamic Forecast: This forecast automatically updates based on your payables selections. Selected payments will impact cash flow in their scheduled payment weeks.")

    # Show table (formatted)
    show = table.copy()
    for c in week_cols:
        show[c] = show[c].apply(money)
    st.dataframe(show, use_container_width=True, height=520)

    # Download
    buf = io.BytesIO()
    table.to_excel(buf, index=False, sheet_name="13-Week Forecast")
    buf.seek(0)
    st.download_button("Download 13-Week Forecast (.xlsx)", data=buf,
                       file_name="cash_forecast_13w.xlsx", use_container_width=True)

    # Trend chart
    st.markdown("#### Ending Cash Trend (13 Weeks)")
    trend = pd.DataFrame({
        "Week": list(range(1,14)),
        "EndingCash": end_series.tolist()
    })
    st.line_chart(trend, x="Week", y="EndingCash", height=220, use_container_width=True)

# -----------------------------
# Footer & Template Downloads
# -----------------------------
st.divider()
t1, t2 = st.columns(2)
with t1:
    st.download_button("Download GL (90-day) Template (.xlsx)", data=template_bytes("gl"),
                       file_name="gl_90d_template.xlsx", use_container_width=True)
with t2:
    st.markdown(
        """
        <div style="font-size:12px;color:#6b7280;">
        Notes: A/P & A/R are bucketed into weeks by <em>DueDate</em>. GL is used (optionally) to estimate weekly Operating Expenses and Payroll from recent history (simple heuristics).
        </div>
        """, unsafe_allow_html=True
    )
