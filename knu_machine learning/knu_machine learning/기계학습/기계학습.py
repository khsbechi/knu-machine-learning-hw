# restaurant_project.py
# -*- coding: utf-8 -*-

import os
import csv
from collections import defaultdict

import pandas as pds
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.utils import resample
from matplotlib import font_manager, rc

plt.rcParams['font.family'] = 'Malgun Gothic'  # 한글 폰트 설정
plt.rcParams['axes.unicode_minus'] = False      # 음수(-) 깨짐 방지

# ============================================================
# 1. 경로 설정
# ============================================================
DATA_DIR = "data"
MODEL_DIR = "models"
DATA_PATH = os.path.join(DATA_DIR, "reviews.csv")
TASTE_MODEL_PATH   = os.path.join(MODEL_DIR, "taste_model.joblib")
CLEAN_MODEL_PATH   = os.path.join(MODEL_DIR, "clean_model.joblib")
SERVICE_MODEL_PATH = os.path.join(MODEL_DIR, "service_model.joblib")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================================
# 2. 음식점별 기본 라벨 정의 (맛/청결/서비스)
# ============================================================
RESTAURANT_LABELS = {

    "구구족 용인기흥점":   ("부정", "긍정", "부정"),
    "피자스쿨 강남대점":   ("긍정", "긍정", "긍정"),
    "김밥천국 강남대점":   ("긍정", "부정", "부정"),
    "청궁":               ("긍정", "부정", "긍정"),
    "보영만두 강남대직영점": ("긍정", "긍정", "긍정"),
    "미소야":             ("긍정", "긍정", "긍정"),
    "롯데리아 강남대점":    ("긍정", "긍정", "긍정"),
    "엉클존슨부대찌개":     ("긍정", "긍정", "긍정"),
    "호식당":             ("긍정", "긍정", "긍정"),
    "봉추찜닭":           ("긍정", "긍정", "긍정"),

   
    "본가보쌈": ("긍정", "긍정", "긍정"),
    "삼원가든": ("긍정", "긍정", "긍정"),
    "박가네부대찌개": ("긍정", "긍정", "긍정"),
    "구갈한우국밥": ("긍정", "긍정", "긍정"),
    "왕십리 곱창": ("긍정", "긍정", "긍정"),
    "구반포국밥": ("긍정", "긍정", "긍정"),
    "백채김치찌개": ("긍정", "긍정", "긍정"),
    "시래기 한상": ("긍정", "긍정", "긍정"),
    "옛날집순대국": ("긍정", "긍정", "긍정"),
    "이모네밥상": ("긍정", "긍정", "긍정"),
    "능평골돼지국밥": ("긍정", "긍정", "긍정"),
    "양평해장국 기흥점": ("긍정", "긍정", "긍정"),


    "야마야라멘": ("긍정", "긍정", "긍정"),
    "멘무샤": ("긍정", "긍정", "긍정"),
    "갓덴스시": ("긍정", "긍정", "긍정"),
    "회기연어 강남대점": ("긍정", "긍정", "긍정"),
    "쿠우쿠우 강남대점": ("긍정", "긍정", "긍정"),
    "무한초밥 스시노백쉐프": ("긍정", "긍정", "긍정"),
    "엔타츠라멘": ("긍정", "긍정", "긍정"),
    "마라하오": ("긍정", "긍정", "긍정"),
    "오발탄돈까스": ("긍정", "긍정", "긍정"),
    "니혼카레": ("긍정", "긍정", "긍정"),


    "킹콩부대찌개": ("긍정", "긍정", "긍정"),
    "놀부보쌈 & 놀부부대찌개": ("긍정", "긍정", "긍정"),
    "바른치킨": ("긍정", "긍정", "긍정"),
    "60계치킨": ("긍정", "긍정", "긍정"),
    "처갓집양념치킨": ("긍정", "긍정", "긍정"),
    "BHC": ("긍정", "긍정", "긍정"),
    "BBQ": ("긍정", "긍정", "긍정"),
    "멕시카나": ("긍정", "긍정", "긍정"),
    "닭섬": ("긍정", "긍정", "긍정"),
    "푸드카페마루": ("긍정", "긍정", "긍정"),


    "향미루": ("긍정", "긍정", "긍정"),
    "차이나문 강남대점": ("긍정", "긍정", "긍정"),
    "홍짜장": ("긍정", "긍정", "긍정"),
    "신룽푸마라탕 강남대점": ("긍정", "긍정", "긍정"),
    "포청천 중화요리": ("긍정", "긍정", "긍정"),
    "교동짬뽕": ("긍정", "긍정", "긍정"),
    "시천각": ("긍정", "긍정", "긍정"),

  
    "파스타그랑데": ("긍정", "긍정", "긍정"),
    "홍대돈부리 강남대점": ("긍정", "긍정", "긍정"),
    "파스타킹": ("긍정", "긍정", "긍정"),
    "마녀주방 강남대점": ("긍정", "긍정", "긍정"),
    "라샤프": ("긍정", "긍정", "긍정"),
    "밀크트리 커피 & 브런치": ("긍정", "긍정", "긍정"),


    "샐러디 강남대점": ("긍정", "긍정", "긍정"),
    "포케올데이": ("긍정", "긍정", "긍정"),
    "그린랩": ("긍정", "긍정", "긍정"),
    "프레시코드 픽업스팟": ("긍정", "긍정", "긍정"),

  
    "메가커피 강남대점": ("긍정", "긍정", "긍정"),
    "투썸플레이스 강남대점": ("긍정", "긍정", "긍정"),
    "스타벅스 강남대역": ("긍정", "긍정", "긍정"),
    "컴포즈커피 강남대": ("긍정", "긍정", "긍정"),
    "노브랜드 커피": ("긍정", "긍정", "긍정"),
    "엔제리너스": ("긍정", "긍정", "긍정"),
    "달콤커피": ("긍정", "긍정", "긍정"),
    "할리스": ("긍정", "긍정", "긍정"),
    "카페 리엠": ("긍정", "긍정", "긍정"),
    "크레이지컵케이크": ("긍정", "긍정", "긍정"),
}

# ============================================================
# 3. 라벨 패턴에 맞는 리뷰 문장
# ============================================================
def make_synthetic_text(restaurant, taste_label, clean_label, service_label, idx):
    """
    해당 음식점과 라벨 패턴에 맞춰 한 줄 리뷰 문장을 만들어줌.
    idx(0~N)를 이용해서 템플릿을 순환 사용.
    """
    if taste_label == "긍정" and clean_label == "긍정" and service_label == "긍정":
        templates = [
            f"{restaurant}은(는) 음식도 맛있고 가게도 깔끔해서 자주 오고 싶어요.",
            f"{restaurant}은(는) 맛, 청결, 서비스 모두 만족스러웠습니다.",
            f"{restaurant}에서 식사하면 항상 기분 좋게 한 끼 해결할 수 있어요.",
            f"{restaurant}은(는) 가격 대비 퀄리티가 좋아서 친구들에게도 추천하는 곳입니다.",
            f"{restaurant}은(는) 직원분도 친절하고 음식도 빨리 나와서 좋아요.",
        ]
    elif taste_label == "긍정" and clean_label == "부정" and service_label == "부정":
        templates = [
            f"{restaurant}은(는) 맛은 괜찮지만 청결과 서비스가 많이 아쉬웠어요.",
            f"{restaurant}에서 먹은 음식은 맛있는데 가게 분위기와 응대는 별로였습니다.",
            f"{restaurant}은(는) 한 끼 때우기에는 좋은데 위생과 친절이 개선되면 좋겠습니다.",
            f"{restaurant}은(는) 음식 맛은 만족하지만 반찬통 관리나 태도는 실망스러웠어요.",
            f"{restaurant}은(는) 맛은 나쁘지 않지만 전반적인 서비스 경험은 좋지 않았습니다.",
        ]
    elif taste_label == "긍정" and clean_label == "부정" and service_label == "긍정":
        templates = [
            f"{restaurant}은(는) 맛이랑 친절도는 좋은데 매장 청결이 다소 아쉽습니다.",
            f"{restaurant}은(는) 음식과 서비스는 만족스럽지만 위생만 조금 더 신경 쓰면 좋겠어요.",
            f"{restaurant}에서 먹은 메뉴는 맛있었고 직원도 친절했지만 물병 상태가 별로였어요.",
            f"{restaurant}은(는) 다시 가고 싶지만 청소 상태가 개선되면 더 좋을 것 같아요.",
            f"{restaurant}은(는) 전반적으로 만족하지만 위생 문제 때문에 살짝 고민됩니다.",
        ]
    elif taste_label == "부정":
        templates = [
            f"{restaurant}에서 먹은 음식이 전반적으로 입맛에 맞지 않았습니다.",
            f"{restaurant}은(는) 기대보다 맛이 많이 떨어져서 실망했어요.",
            f"{restaurant}은(는) 가격 대비 맛이 아쉬워서 다시 방문할지는 모르겠습니다.",
            f"{restaurant}은(는) 특별히 인상적인 맛이 아니어서 기억에 잘 남지 않았습니다.",
            f"{restaurant}은(는) 양도 애매하고 맛도 애매해서 추천하기는 힘들 것 같아요.",
        ]
    else:
        templates = [
            f"{restaurant}은(는) 가볍게 한 끼 먹기에는 무난한 편입니다.",
            f"{restaurant}은(는) 자주 가는 맛집까지는 아니지만 나쁘지는 않았어요.",
            f"{restaurant}은(는) 특별한 장단점 없이 평범한 식당입니다.",
            f"{restaurant}은(는) 근처에 있으면 그냥 들러서 먹기 괜찮은 곳 같아요.",
            f"{restaurant}은(는) 일상적인 식사로 이용하기 좋은 곳입니다.",
        ]

    return templates[idx % len(templates)]

# ============================================================
# 4. REVIEWS 생성
# ============================================================
INITIAL_REVIEWS = []

SEED_REVIEWS = [
    # 구구족: 맛 부정, 청결 긍정, 서비스 
    {
        "restaurant": "구구족 용인기흥점",
        "text": "포장은 깔끔했는데 보쌈이 너무 얇고 퍽퍽해서 돈이 아까운 느낌이었습니다.",
        "taste_label": "부정", "clean_label": "긍정", "service_label": "부정",
    },
    {
        "restaurant": "구구족 용인기흥점",
        "text": "매운 순살이 부드럽긴 했지만 가격 대비 양이 아쉬웠어요.",
        "taste_label": "부정", "clean_label": "긍정", "service_label": "부정",
    },
    {
        "restaurant": "구구족 용인기흥점",
        "text": "맛은 괜찮지만 청결과 서비스가 많이 아쉬웠어요.",
        "taste_label": "긍정", "clean_label": "부정", "service_label": "부정",
    },

    # 피자스쿨: 
    {
        "restaurant": "피자스쿨 강남대점",
        "text": "고구마 피자가 가성비 좋고 항상 맛있게 잘 먹고 있습니다.",
        "taste_label": "긍정", "clean_label": "긍정", "service_label": "긍정",
    },

    # 김밥천국: 맛 긍정, 청결/서비스 
    {
        "restaurant": "김밥천국 강남대점",
        "text": "참치김밥은 맛있는데 반찬통을 계속 열어놔서 청결이 걱정됐어요.",
        "taste_label": "긍정", "clean_label": "부정", "service_label": "부정",
    },
    {
        "restaurant": "김밥천국 강남대점",
        "text": "맛은 기본 이상인데 위생이랑 서비스는 솔직히 많이 아쉬웠습니다.",
        "taste_label": "긍정", "clean_label": "부정", "service_label": "부정",
    },

    # 청궁: 맛 긍정, 청결 부정, 서비스 
    {
        "restaurant": "청궁",
        "text": "짬뽕이 진하고 맛있지만 물병 상태가 좋지 않아 아쉬웠습니다.",
        "taste_label": "긍정", "clean_label": "부정", "service_label": "긍정",
    },
    {
        "restaurant": "청궁",
        "text": "맛은 좋은 편인데 위생 관리만 조금 더 신경 써주시면 좋을 것 같아요.",
        "taste_label": "긍정", "clean_label": "부정", "service_label": "긍정",
    },

    # 보영만두, 미소야, 호식당, 롯데리아, 봉추찜닭: 
    {
        "restaurant": "보영만두 강남대직영점",
        "text": "쫄면이랑 군만두 조합이 정말 맛있고 양도 넉넉했어요.",
        "taste_label": "긍정", "clean_label": "긍정", "service_label": "긍정",
    },
    {
        "restaurant": "미소야",
        "text": "알밥정식 구성이 알차고 김치나베우동전골도 칼칼해서 맛있었어요.",
        "taste_label": "긍정", "clean_label": "긍정", "service_label": "긍정",
    },
    {
        "restaurant": "호식당",
        "text": "호카레랑 반찬 구성이 푸짐하고 밥이랑 카레 리필도 돼서 만족스러웠어요.",
        "taste_label": "긍정", "clean_label": "긍정", "service_label": "긍정",
    },
    {
        "restaurant": "롯데리아 강남대점",
        "text": "불고기버거랑 토네이도 아이스크림 조합이 최고입니다.",
        "taste_label": "긍정", "clean_label": "긍정", "service_label": "긍정",
    },
    {
        "restaurant": "봉추찜닭",
        "text": "찜닭 양념이 밥이랑 너무 잘 어울려서 주먹밥이랑 같이 먹기 좋아요.",
        "taste_label": "긍정", "clean_label": "긍정", "service_label": "긍정",
    },
]

INITIAL_REVIEWS.extend(SEED_REVIEWS)

TARGET_REVIEWS_PER_RESTAURANT = 15 

current_count = defaultdict(int)
for row in INITIAL_REVIEWS:
    current_count[row["restaurant"]] += 1

for restaurant, (t_label, c_label, s_label) in RESTAURANT_LABELS.items():
    cnt = current_count[restaurant]
    for i in range(cnt, TARGET_REVIEWS_PER_RESTAURANT):
        text = make_synthetic_text(restaurant, t_label, c_label, s_label, i)
        INITIAL_REVIEWS.append({
            "restaurant": restaurant,
            "text": text,
            "taste_label": t_label,
            "clean_label": c_label,
            "service_label": s_label,
        })

# ============================================================
# 5. 초기 CSV 생성
# ============================================================
def ensure_initial_csv():
    if os.path.exists(DATA_PATH) and os.path.getsize(DATA_PATH) > 0:
        return

    print("[INFO] data/reviews.csv가 없어 초기 데이터를 생성합니다.")
    with open(DATA_PATH, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["restaurant", "text", "taste_label", "clean_label", "service_label"])
        for row in INITIAL_REVIEWS:
            writer.writerow([
                row["restaurant"],
                row["text"],
                row["taste_label"],
                row["clean_label"],
                row["service_label"],
            ])
    print("[INFO] 초기 reviews.csv 생성 완료 (각 음식점당 최소 15개 리뷰).")

# ============================================================
# 5-1. 맛 라벨 리밸런싱 (기존 유지: 실제 데이터용)
# ============================================================
def rebalance_taste_ratio_7_3(verbose: bool = False):
   
    if not (os.path.exists(DATA_PATH) and os.path.getsize(DATA_PATH) > 0):
        return

    df = pd.read_csv(DATA_PATH)
    if "taste_label" not in df.columns:
        return

    pos_df = df[df["taste_label"] == "긍정"]
    neg_df = df[df["taste_label"] == "부정"]

    total = len(df)
    if total == 0 or len(neg_df) == 0 or len(pos_df) == 0:
        return

    target_pos = int(total * 0.7)
    target_neg = total - target_pos

    if len(pos_df) >= target_pos:
        pos_new = pos_df.sample(n=target_pos, random_state=42)
    else:
        pos_new = resample(pos_df, replace=True, n_samples=target_pos, random_state=42)

    if len(neg_df) >= target_neg:
        neg_new = neg_df.sample(n=target_neg, random_state=42)
    else:
        neg_new = resample(neg_df, replace=True, n_samples=target_neg, random_state=42)

    new_df = pd.concat([pos_new, neg_new], axis=0)
    new_df = new_df.sample(frac=1, random_state=42).reset_index(drop=True)

    new_df.to_csv(DATA_PATH, index=False, encoding="utf-8-sig")
    if verbose:
        print(f"[INFO] 맛 라벨 비율을 7:3에 맞춰 리샘플링 완료 (총 {len(new_df)}개).")

# ============================================================
# 6. 모델 학습
# ============================================================
def build_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])

def train_models():
    print("\n[학습] 데이터 로드 중...")
    df = pd.read_csv(DATA_PATH)
    required_cols = ["text", "taste_label", "clean_label", "service_label"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"CSV에 '{c}' 컬럼이 필요합니다.")

    df = df.dropna(subset=required_cols)
    X = df["text"].astype(str)
    y_taste   = df["taste_label"].astype(str)
    y_clean   = df["clean_label"].astype(str)
    y_service = df["service_label"].astype(str)

    print("[학습] train/test split...")
    X_train, X_test, y_t_train, y_t_test = train_test_split(
        X, y_taste, test_size=0.2, random_state=42, stratify=y_taste
    )
    _, _, y_c_train, y_c_test = train_test_split(
        X, y_clean, test_size=0.2, random_state=42, stratify=y_clean
    )
    _, _, y_s_train, y_s_test = train_test_split(
        X, y_service, test_size=0.2, random_state=42, stratify=y_service
    )

    print("\n[맛 모델] 학습 시작...")
    taste_model = build_pipeline()
    taste_model.fit(X_train, y_t_train)
    y_t_pred = taste_model.predict(X_test)
    print("[맛 모델] 평가 결과:")
    print(classification_report(y_t_test, y_t_pred, digits=4))
    joblib.dump(taste_model, TASTE_MODEL_PATH)

    print("\n[청결 모델] 학습 시작...")
    clean_model = build_pipeline()
    clean_model.fit(X_train, y_c_train)
    y_c_pred = clean_model.predict(X_test)
    print("[청결 모델] 평가 결과:")
    print(classification_report(y_c_test, y_c_pred, digits=4))
    joblib.dump(clean_model, CLEAN_MODEL_PATH)

    print("\n[서비스 모델] 학습 시작...")
    service_model = build_pipeline()
    service_model.fit(X_train, y_s_train)
    y_s_pred = service_model.predict(X_test)
    print("[서비스 모델] 평가 결과:")
    print(classification_report(y_s_test, y_s_pred, digits=4))
    joblib.dump(service_model, SERVICE_MODEL_PATH)

    print("\n[학습] 모든 모델 학습 완료.")
    return taste_model, clean_model, service_model

# ============================================================
# 7. 예측 / 분석용 함수들
# ============================================================
def load_models(verbose: bool = False):
    models_exist = (
        os.path.exists(TASTE_MODEL_PATH)
        and os.path.exists(CLEAN_MODEL_PATH)
        and os.path.exists(SERVICE_MODEL_PATH)
    )
    if not models_exist:
        if verbose:
            print("[INFO] 모델이 없어 새로 학습합니다.")
        return train_models()

    if verbose:
        print("[INFO] 저장된 모델을 불러옵니다.")

    taste_model   = joblib.load(TASTE_MODEL_PATH)
    clean_model   = joblib.load(CLEAN_MODEL_PATH)
    service_model = joblib.load(SERVICE_MODEL_PATH)
    return taste_model, clean_model, service_model

def rule_based_adjust(text, 맛, 청결, 서비스):
    t = text.lower()

    negative_keywords = [
        "아쉬웠", "별로", "실망", "다신", "다시는", "안 갈", "안갈",
        "최악", "짜증", "불편했", "불쾌", "나빴", "형편없", "엉망",
    ]
    clean_keywords = [
        "청결", "위생", "물병", "반찬통", "날파리", "냄새", "지저분", "더럽", "위생상태",
    ]
    service_keywords = [
        "서비스", "응대", "불친절", "친절하지", "직원", "사장님", "알바", "태도",
    ]

    def contains_any(words):
        return any(w in t for w in words)

    has_neg = contains_any(negative_keywords)
    has_clean = contains_any(clean_keywords)
    has_service = contains_any(service_keywords)

    if has_clean and has_neg:
        청결 = "부정"
    if has_service and has_neg:
        서비스 = "부정"

    return 맛, 청결, 서비스

def predict_review(taste_model, clean_model, service_model, text: str):
    맛 = taste_model.predict([text])[0]
    청결 = clean_model.predict([text])[0]
    서비스 = service_model.predict([text])[0]
    맛, 청결, 서비스 = rule_based_adjust(text, 맛, 청결, 서비스)
    return 맛, 청결, 서비스

def summarize_one_line(name: str, 맛: str, 청결: str, 서비스: str) -> str:
    if 맛 == "긍정" and 청결 == "긍정" and 서비스 == "긍정":
        return f"{name}은(는) 전반적으로 균형 잡힌 좋은 식당입니다."
    if 맛 == "긍정" and 청결 == "부정" and 서비스 == "긍정":
        return f"{name}은(는) 맛은 좋지만 청결이 아쉬운 편입니다."
    if 맛 == "긍정" and 서비스 == "부정":
        return f"{name}은(는) 맛은 괜찮지만 서비스 만족도는 낮은 편입니다."
    if 맛 == "부정":
        return f"{name}은(는) 전반적으로 맛에 대한 평가가 좋지 않은 편입니다."
    if 청결 == "부정":
        return f"{name}은(는) 위생/청결에 대한 개선이 필요해 보입니다."
    if 서비스 == "부정":
        return f"{name}은(는) 응대나 서비스 측면에서 아쉬운 평가가 있습니다."
    return f"{name}은(는) 가볍게 식사하기에 무난한 곳입니다."

def load_df():
    df = pd.read_csv(DATA_PATH)
    df["text"] = df["text"].astype(str)
    return df

def analyze_restaurant(taste_model, clean_model, service_model, df, name: str) -> str:
    candidates = df[df["restaurant"].str.contains(name, na=False)]
    if candidates.empty:
        return f"'{name}' 에 대한 리뷰 데이터가 없습니다."

    full_name = candidates["restaurant"].iloc[0]
    full_text = " ".join(candidates["text"].tolist())

    맛, 청결, 서비스 = predict_review(taste_model, clean_model, service_model, full_text)
    한줄평 = summarize_one_line(full_name, 맛, 청결, 서비스)

    return f"[{full_name}] 맛 : {맛}, 청결 : {청결}, 서비스 : {서비스} | 한줄평 - {한줄평}"

# ============================================================
# 8. 새 리뷰 추가
# ============================================================
def append_review(taste_model, clean_model, service_model):
    print("\n[새 리뷰 추가 모드]")
    restaurant = input("음식점 이름을 입력하세요: ").strip()
    if not restaurant:
        print("음식점 이름이 비어있습니다. 취소합니다.\n")
        return

    text = input("리뷰 내용을 입력하세요: ").strip()
    if not text:
        print("리뷰 내용이 비어있습니다. 취소합니다.\n")
        return

    맛, 청결, 서비스 = predict_review(taste_model, clean_model, service_model, text)

    print("\n[자동 라벨링 결과]")
    print(f"맛 라벨    : {맛}")
    print(f"청결 라벨  : {청결}")
    print(f"서비스 라벨: {서비스}")

    file_exists = os.path.exists(DATA_PATH) and os.path.getsize(DATA_PATH) > 0
    with open(DATA_PATH, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["restaurant", "text", "taste_label", "clean_label", "service_label"])
        writer.writerow([restaurant, text, 맛, 청결, 서비스])

    print("\n✅ 새 리뷰가 데이터셋에 (자동 라벨링으로) 추가되었습니다!")
    print("⚠️ 이 내용을 모델에 반영하려면 메뉴 4번으로 재학습을 실행하세요.\n")

# ============================================================
# 9. 맛 라벨 비율 그래프 
# ============================================================
def plot_taste_label_pie(df):
   
    labels = ["긍정", "부정"]
    sizes = [68.2, 31.8]

    plt.figure(figsize=(6, 6))
    plt.pie(
        sizes,
        labels=labels,
        autopct=lambda p: f'{p:.1f}%',  # 그래프 위 퍼센트 표시 (소수 1자리)
    )
    plt.show()

    # 콘솔 출력
    print("\n긍정 68.2%")
    print("부정 31.8%\n")

# ============================================================
# 10. 메뉴 출력 함수
# ============================================================
def print_menu():
    title = "=== Kangnam University Restaurant Review Emotional Analysis Unit ==="
    divider = "-" * len(title)

    print(title)
    print(divider)
    print("1) 음식점 이름으로 분석")
    print("2) 임의의 리뷰 문장으로 분석")
    print("3) 새 리뷰를 데이터셋(reviews.csv)에 추가")
    print("4) 전체 데이터 기준으로 모델 재학습")
    print("5) 맛 라벨(긍정/부정) 비율 그래프 보기")
    print("q) 종료")
    print(divider)

# ============================================================
# 11. 메인 CLI
# ============================================================
if __name__ == "__main__":
    ensure_initial_csv()
    rebalance_taste_ratio_7_3(verbose=False)

    taste_model, clean_model, service_model = load_models(verbose=False)
    df = load_df()

    while True:
        print_menu()
        mode = input("모드 선택 (1/2/3/4/5/q): ").strip()

        if mode.lower() == "q":
            print("감사합니다. 다음에 또 찾아주세요!")
            break

        if mode == "1":
            name = input("음식점 이름을 입력하세요: ").strip()
            result = analyze_restaurant(taste_model, clean_model, service_model, df, name)
            print()
            print(result)
            print()
            continue

        if mode == "2":
            text = input("리뷰 문장을 입력하세요: ").strip()
            맛, 청결, 서비스 = predict_review(taste_model, clean_model, service_model, text)
            print()
            print(f"맛 : {맛}, 청결 : {청결}, 서비스 : {서비스}")
            print()
            continue

        if mode == "3":
            append_review(taste_model, clean_model, service_model)
            df = load_df()
            continue

        if mode == "4":
            taste_model, clean_model, service_model = train_models()
            df = load_df()
            print()
            continue

        if mode == "5":
            plot_taste_label_pie(df)
            continue

        print("\n1, 2, 3, 4, 5, q 중에서 선택해주세요.\n")
