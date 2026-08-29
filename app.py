import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="ChargebackIQ", page_icon="🛡️", layout="centered")

with open('logreg_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('feature_cols.pkl', 'rb') as f:
    feature_cols = pickle.load(f)

LOW_THRESHOLD = 0.30
HIGH_THRESHOLD = 0.75

def tier_decision(prob, low=LOW_THRESHOLD, high=HIGH_THRESHOLD):
    if prob >= high:
        return 'Auto-Approve'
    elif prob < low:
        return 'Auto-Reject'
    else:
        return 'Flag for Human Review'

def generate_evidence_packet(row, prob, tier):
    lines = []
    lines.append(f"CHARGEBACK EVIDENCE PACKET — Transaction {row['transaction_id']}")
    lines.append(f"Dispute Reason: {row['chargeback_reason'].replace('_',' ').title()}")
    lines.append(f"Transaction Amount: ₹{row['amount']:,.2f}")
    lines.append("")
    lines.append("EVIDENCE SUMMARY:")
    lines.append(f"  {'✓' if row['delivery_confirmed'] else '✗'} Delivery {'confirmed within ' + str(row['delivery_days']) + ' days' if row['delivery_confirmed'] else 'not confirmed'}.")
    lines.append(f"  {'✓' if row['ip_geo_match'] else '✗'} IP geolocation {'matches' if row['ip_geo_match'] else 'does not match'} billing address.")
    lines.append(f"  {'✓' if row['avs_match'] else '✗'} AVS {'matched' if row['avs_match'] else 'did not match'}.")
    lines.append(f"  {'✓' if row['cvv_match'] else '✗'} CVV verification {'passed' if row['cvv_match'] else 'failed'}.")
    lines.append(f"  Customer has {row['prior_orders_count']} prior order(s) and {row['customer_dispute_history_count']} prior dispute(s).")
    lines.append("")
    lines.append(f"MODEL ASSESSMENT: Win probability = {prob:.1%}  |  Recommended action: {tier}")
    lines.append("")
    lines.append("NOTE: This packet reflects only verified transaction fields. No claims beyond recorded evidence are made.")
    return "\n".join(lines)

st.title("🛡️ ChargebackIQ")
st.markdown("Predict dispute win-probability, auto-generate evidence packets, and route uncertain cases to human review. **Defense-only** — no autonomous financial actions.")

st.divider()
st.subheader("Enter Transaction Evidence")

col1, col2 = st.columns(2)
with col1:
    amount = st.number_input("Transaction Amount (₹)", min_value=1.0, value=1500.0)
    reason = st.selectbox("Chargeback Reason", ['item_not_received', 'unauthorized', 'not_as_described'])
    delivery_confirmed = st.checkbox("Delivery Confirmed", value=True)
    delivery_days = st.slider("Delivery Days", 1, 15, 5)
with col2:
    ip_geo_match = st.checkbox("IP Geolocation Match", value=True)
    avs_match = st.checkbox("AVS Match", value=True)
    cvv_match = st.checkbox("CVV Match", value=True)
    prior_orders = st.number_input("Prior Orders Count", min_value=0, value=3)
    prior_disputes = st.number_input("Prior Dispute History Count", min_value=0, value=0)

days_since_signup = st.slider("Days Since Signup", 1, 1800, 365)

if st.button("🔍 Analyze Dispute", type="primary"):
    row = pd.Series({
        'transaction_id': 'TXN-DEMO-001',
        'amount': amount,
        'chargeback_reason': reason,
        'delivery_confirmed': delivery_confirmed,
        'delivery_days': delivery_days,
        'ip_geo_match': ip_geo_match,
        'avs_match': avs_match,
        'cvv_match': cvv_match,
        'prior_orders_count': prior_orders,
        'customer_dispute_history_count': prior_disputes,
        'days_since_signup': days_since_signup,
    })

    X = pd.DataFrame([{
        'delivery_confirmed': int(delivery_confirmed),
        'ip_geo_match': int(ip_geo_match),
        'avs_match': int(avs_match),
        'cvv_match': int(cvv_match),
        'prior_orders_count': prior_orders,
        'days_since_signup': days_since_signup,
        'customer_dispute_history_count': prior_disputes,
        'amount': amount,
        'delivery_days': delivery_days,
        'reason_item_not_received': 1 if reason=='item_not_received' else 0,
        'reason_not_as_described': 1 if reason=='not_as_described' else 0,
        'reason_unauthorized': 1 if reason=='unauthorized' else 0,
    }])
    correct_order = ['delivery_confirmed','ip_geo_match','avs_match','cvv_match',
                      'prior_orders_count','days_since_signup','customer_dispute_history_count',
                      'amount','delivery_days','reason_item_not_received',
                      'reason_not_as_described','reason_unauthorized']
    X = X[correct_order]
    X_scaled = scaler.transform(X)
    prob = model.predict_proba(X_scaled)[0][1]
    tier = tier_decision(prob)

    st.divider()
    c1, c2 = st.columns(2)
    c1.metric("Win Probability", f"{prob:.1%}")
    color = {"Auto-Approve":"🟢", "Flag for Human Review":"🟡", "Auto-Reject":"🔴"}
    c2.metric("Recommended Action", f"{color[tier]} {tier}")

    st.subheader("Generated Evidence Packet")
    packet = generate_evidence_packet(row, prob, tier)
    st.code(packet, language=None)

    st.caption("⚠️ Defense-only system. This tool evaluates and documents evidence; it takes no autonomous financial action.")
