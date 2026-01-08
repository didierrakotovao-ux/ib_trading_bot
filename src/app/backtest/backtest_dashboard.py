"""
Dashboard Streamlit pour visualiser les résultats du backtest.
Lancer avec: streamlit run backtest_dashboard.py
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
from datetime import datetime
import glob
import os

st.set_page_config(
    page_title="Backtest Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📊 Dashboard Backtest Trading")


def load_latest_journal():
    """Charge le dernier journal de trading."""
    journals = glob.glob("trading_journal_*.csv")
    if not journals:
        return None
    latest = max(journals, key=os.path.getctime)
    df = pd.read_csv(latest, parse_dates=['date_entree', 'date_sortie'])
    # Compatibilité avec l'ancien format (pnl) et nouveau format (pnl_net)
    if 'pnl_net' in df.columns:
        df['pnl'] = df['pnl_net']
    return df


def load_latest_results():
    """Charge les derniers résultats du backtest."""
    results_files = glob.glob("backtest_results_*.json")
    if not results_files:
        return None
    latest = max(results_files, key=os.path.getctime)
    with open(latest, 'r') as f:
        return json.load(f)


def calculate_equity_curve(df, initial_capital=100000):
    """Calcule la courbe d'équité à partir des trades."""
    if df.empty:
        return pd.DataFrame()

    df = df.sort_values('date_sortie')
    df['cumulative_pnl'] = df['pnl'].cumsum()
    df['equity'] = initial_capital + df['cumulative_pnl']
    return df


def main():
    # Charger les données
    journal_df = load_latest_journal()
    results = load_latest_results()

    if journal_df is None or journal_df.empty:
        st.warning("⚠️ Aucun journal de trading trouvé. Lancez d'abord un backtest.")
        st.info("Le fichier doit être nommé: trading_journal_YYYYMMDD_HHMMSS.csv")
        return

    # ==========================================
    # SECTION 1: MÉTRIQUES CLÉS
    # ==========================================
    st.header("📈 Métriques Clés")

    total_trades = len(journal_df)
    winning_trades = len(journal_df[journal_df['pnl'] > 0])
    losing_trades = len(journal_df[journal_df['pnl'] < 0])
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

    total_pnl = journal_df['pnl'].sum()
    avg_win = journal_df[journal_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
    avg_loss = journal_df[journal_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
    profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if losing_trades > 0 and avg_loss != 0 else 0

    # Affichage des métriques en colonnes
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("PnL Total", f"${total_pnl:,.2f}",
                  delta=f"{total_pnl/1000:.1f}%" if results else None)
    with col2:
        st.metric("Win Rate", f"{win_rate:.1f}%")
    with col3:
        st.metric("Trades", f"{total_trades}")
    with col4:
        st.metric("Gain Moyen", f"${avg_win:,.2f}" if avg_win else "$0")
    with col5:
        st.metric("Perte Moyenne", f"${avg_loss:,.2f}" if avg_loss else "$0")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Gagnants", f"{winning_trades}", delta=None)
    with col2:
        st.metric("Perdants", f"{losing_trades}", delta=None)
    with col3:
        st.metric("Profit Factor", f"{profit_factor:.2f}")
    with col4:
        if results:
            sharpe = results.get('sharpe', {}).get('sharperatio')
            st.metric("Sharpe Ratio", f"{sharpe:.2f}" if sharpe else "N/A")
        else:
            st.metric("Sharpe Ratio", "N/A")
    with col5:
        if results:
            dd = results.get('drawdown', {}).get('max', {}).get('drawdown', 0)
            st.metric("Max Drawdown", f"{dd:.2f}%")
        else:
            max_dd = calculate_max_drawdown(journal_df)
            st.metric("Max Drawdown", f"{max_dd:.2f}%")

    st.divider()

    # ==========================================
    # SECTION 2: GRAPHIQUES
    # ==========================================
    st.header("📉 Visualisations")

    tab1, tab2, tab3, tab4 = st.tabs(["Courbe d'Équité", "Distribution PnL", "Par Symbole", "Timeline"])

    with tab1:
        # Courbe d'équité
        equity_df = calculate_equity_curve(journal_df)
        if not equity_df.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=equity_df['date_sortie'],
                y=equity_df['equity'],
                mode='lines+markers',
                name='Équité',
                line=dict(color='#00CC96', width=2),
                marker=dict(size=6)
            ))
            fig.add_hline(y=100000, line_dash="dash", line_color="gray",
                         annotation_text="Capital Initial")
            fig.update_layout(
                title="Évolution du Capital",
                xaxis_title="Date",
                yaxis_title="Équité ($)",
                hovermode='x unified',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        # Distribution des PnL
        col1, col2 = st.columns(2)

        with col1:
            # Histogramme des PnL
            fig = px.histogram(
                journal_df,
                x='pnl',
                nbins=20,
                color_discrete_sequence=['#636EFA'],
                title="Distribution des Profits/Pertes"
            )
            fig.add_vline(x=0, line_dash="dash", line_color="red")
            fig.update_layout(xaxis_title="PnL ($)", yaxis_title="Nombre de trades")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Pie chart Gagnants vs Perdants
            fig = go.Figure(data=[go.Pie(
                labels=['Gagnants', 'Perdants'],
                values=[winning_trades, losing_trades],
                hole=.4,
                marker_colors=['#00CC96', '#EF553B']
            )])
            fig.update_layout(title="Répartition Gagnants/Perdants")
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        # Performance par symbole
        symbol_stats = journal_df.groupby('symbole').agg({
            'pnl': ['sum', 'count', 'mean'],
        }).round(2)
        symbol_stats.columns = ['PnL Total', 'Nb Trades', 'PnL Moyen']
        symbol_stats = symbol_stats.sort_values('PnL Total', ascending=False)

        fig = px.bar(
            symbol_stats.reset_index(),
            x='symbole',
            y='PnL Total',
            color='PnL Total',
            color_continuous_scale=['red', 'yellow', 'green'],
            title="Performance par Symbole"
        )
        fig.update_layout(xaxis_title="Symbole", yaxis_title="PnL Total ($)")
        st.plotly_chart(fig, use_container_width=True)

        # Tableau des stats par symbole
        st.dataframe(symbol_stats, use_container_width=True)

    with tab4:
        # Timeline des trades
        fig = go.Figure()

        for _, trade in journal_df.iterrows():
            color = 'green' if trade['pnl'] > 0 else 'red'
            fig.add_trace(go.Scatter(
                x=[trade['date_entree'], trade['date_sortie']],
                y=[trade['symbole'], trade['symbole']],
                mode='lines+markers',
                line=dict(color=color, width=3),
                marker=dict(size=8),
                name=f"{trade['symbole']}: ${trade['pnl']:.0f}",
                hovertemplate=f"<b>{trade['symbole']}</b><br>" +
                             f"Entrée: {trade['date_entree']}<br>" +
                             f"Sortie: {trade['date_sortie']}<br>" +
                             f"PnL: ${trade['pnl']:.2f}<br>" +
                             f"Cause: {trade['cause_sortie']}<extra></extra>"
            ))

        fig.update_layout(
            title="Timeline des Trades",
            xaxis_title="Date",
            yaxis_title="Symbole",
            showlegend=False,
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ==========================================
    # SECTION 3: JOURNAL DÉTAILLÉ
    # ==========================================
    st.header("📋 Journal de Trading")

    # Filtres
    col1, col2, col3 = st.columns(3)
    with col1:
        symbols = ['Tous'] + sorted(journal_df['symbole'].unique().tolist())
        selected_symbol = st.selectbox("Symbole", symbols)
    with col2:
        causes = ['Toutes', 'TP', 'SL']
        selected_cause = st.selectbox("Cause Sortie", causes)
    with col3:
        pnl_filter = st.selectbox("PnL", ['Tous', 'Gagnants', 'Perdants'])

    # Appliquer les filtres
    filtered_df = journal_df.copy()
    if selected_symbol != 'Tous':
        filtered_df = filtered_df[filtered_df['symbole'] == selected_symbol]
    if selected_cause != 'Toutes':
        filtered_df = filtered_df[filtered_df['cause_sortie'] == selected_cause]
    if pnl_filter == 'Gagnants':
        filtered_df = filtered_df[filtered_df['pnl'] > 0]
    elif pnl_filter == 'Perdants':
        filtered_df = filtered_df[filtered_df['pnl'] < 0]

    # Formater le dataframe pour l'affichage
    display_df = filtered_df.copy()
    display_df['prix_entree'] = display_df['prix_entree'].apply(lambda x: f"${x:,.2f}")
    display_df['prix_sortie'] = display_df['prix_sortie'].apply(lambda x: f"${x:,.2f}")

    # Colonnes à afficher selon le format du fichier
    column_config = {
        "symbole": "Symbole",
        "date_entree": st.column_config.DatetimeColumn("Date Entrée", format="YYYY-MM-DD"),
        "quantite": "Qté",
        "prix_entree": "Prix Entrée",
        "date_sortie": st.column_config.DatetimeColumn("Date Sortie", format="YYYY-MM-DD"),
        "prix_sortie": "Prix Sortie",
        "cause_sortie": "Cause",
        "scoring": "Scoring"
    }

    # Nouveau format avec pnl_brut, commission, pnl_net
    if 'pnl_net' in display_df.columns:
        display_df['pnl_brut'] = display_df['pnl_brut'].apply(lambda x: f"${x:,.2f}")
        display_df['commission'] = display_df['commission'].apply(lambda x: f"${x:,.2f}")
        display_df['pnl_net'] = display_df['pnl_net'].apply(lambda x: f"${x:,.2f}")
        column_config["pnl_brut"] = "PnL Brut"
        column_config["commission"] = "Commission"
        column_config["pnl_net"] = "PnL Net"
    else:
        display_df['pnl'] = display_df['pnl'].apply(lambda x: f"${x:,.2f}")
        column_config["pnl"] = "PnL"

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config=column_config
    )

    # Export
    st.download_button(
        label="📥 Télécharger le journal (CSV)",
        data=journal_df.to_csv(index=False).encode('utf-8'),
        file_name=f"trading_journal_export_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )


def calculate_max_drawdown(df, initial_capital=100000):
    """Calcule le drawdown maximum."""
    if df.empty:
        return 0
    equity_df = calculate_equity_curve(df, initial_capital)
    if equity_df.empty:
        return 0
    equity_df['peak'] = equity_df['equity'].cummax()
    equity_df['drawdown'] = (equity_df['peak'] - equity_df['equity']) / equity_df['peak'] * 100
    return equity_df['drawdown'].max()


if __name__ == "__main__":
    main()
