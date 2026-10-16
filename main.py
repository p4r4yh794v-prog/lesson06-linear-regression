import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)
CUTOFF_YEAR = 2025
MIN_OBSERVATION_DAYS = 300

st.set_page_config(page_title="기온 예측기", page_icon="🌡️", layout="wide")
st.title("🌡️ 서울 기온 예측기")
st.caption(
    "서울의 연평균기온 데이터를 이용해 선형 회귀선을 만들고, "
    "선택한 연도의 추정 기온을 계산합니다."
)


@st.cache_data
def load_yearly_data():
    """CSV를 읽고 관측일 수를 기준으로 유효한 연평균기온을 계산합니다."""
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    required_columns = {"날짜", "평균기온"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            "CSV에 필요한 열이 없습니다: " + ", ".join(sorted(missing))
        )

    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")
    df = df.dropna(subset=["날짜", "평균기온"]).copy()
    df["연도"] = df["날짜"].dt.year

    yearly = (
        df.groupby("연도")
        .agg(
            연평균기온=("평균기온", "mean"),
            관측일수=("평균기온", "count"),
        )
        .reset_index()
    )

    # 수업 기준 기간(2025년까지) 및 관측일 300일 이상인 해만 사용
    yearly = yearly[
        (yearly["연도"] <= CUTOFF_YEAR)
        & (yearly["관측일수"] >= MIN_OBSERVATION_DAYS)
    ].sort_values("연도")

    return yearly.reset_index(drop=True)


try:
    yearly = load_yearly_data()
except Exception as exc:
    st.error(f"기온 데이터를 불러오거나 처리하지 못했습니다: {exc}")
    st.info("인터넷 연결과 CSV 주소, 열 이름을 확인한 뒤 다시 실행해 주세요.")
    st.stop()

if len(yearly) < 2:
    st.error("회귀 분석을 하려면 조건을 만족하는 연도가 최소 2개 필요합니다.")
    st.stop()

years = yearly["연도"].to_numpy(dtype=float)
temps = yearly["연평균기온"].to_numpy(dtype=float)

# 1차 다항식(직선) 회귀: 기온 = 기울기 * 연도 + 절편
slope, intercept = np.polyfit(years, temps, 1)
correlation = float(np.corrcoef(years, temps)[0, 1])

selected_year = st.slider(
    "예측할 연도를 선택하세요",
    min_value=1900,
    max_value=2100,
    value=2026,
    step=1,
)
predicted_temp = slope * selected_year + intercept

st.subheader(f"{selected_year}년 예상 연평균기온")
st.metric(
    label="선형 회귀 추정값",
    value=f"{predicted_temp:.2f} °C",
)
st.caption(
    "이 값은 과거 연평균기온의 직선을 연장해 계산한 단순 추정치입니다. "
    "실제 미래 기온이나 공식 기후 전망을 의미하지 않습니다."
)

metric1, metric2, metric3, metric4 = st.columns(4)
metric1.metric("회귀에 사용한 연도 수", f"{len(yearly):,}년")
metric2.metric("시작 연도", f"{int(yearly['연도'].min())}년")
metric3.metric("끝 연도", f"{int(yearly['연도'].max())}년")
metric4.metric("상관계수 (r)", f"{correlation:.4f}")

st.subheader("연도별 평균기온과 회귀 직선")
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["연평균기온"],
        mode="markers",
        name="연평균기온",
        marker={"size": 7},
        customdata=yearly[["관측일수"]].to_numpy(),
        hovertemplate=(
            "연도: %{x}년<br>"
            "연평균기온: %{y:.2f} °C<br>"
            "관측일수: %{customdata[0]}일<extra></extra>"
        ),
    )
)

# 회귀 직선은 실제 학습 데이터의 시작~끝 연도 범위에 표시
line_years = np.linspace(float(years.min()), float(years.max()), 200)
line_temps = slope * line_years + intercept
fig.add_trace(
    go.Scatter(
        x=line_years,
        y=line_temps,
        mode="lines",
        name="선형 회귀 직선",
        line={"width": 3},
        hovertemplate="연도: %{x:.1f}<br>회귀값: %{y:.2f} °C<extra></extra>",
    )
)

fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    hovermode="closest",
    legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
    margin={"l": 20, "r": 20, "t": 30, "b": 20},
)
st.plotly_chart(fig, use_container_width=True)

with st.expander("회귀 분석에 사용한 연도별 데이터 보기"):
    display_data = yearly.copy()
    display_data["연평균기온"] = display_data["연평균기온"].round(2)
    st.dataframe(display_data, use_container_width=True, hide_index=True)

st.caption(
    "데이터 처리 기준: 날짜와 평균기온이 유효한 행만 사용하며, "
    "연도별 유효 관측일 수가 300일 미만이거나 2025년 이후인 해는 제외합니다."
)
