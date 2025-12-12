from __future__ import annotations

import hashlib
import hmac
import json
import logging
import sys
import time
import uuid
import threading
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol

try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings  # type: ignore

try:
    from pydantic import BaseModel, Field
except ImportError:
    raise RuntimeError("pydantic مطلوب لتشغيل SPIG-Core++ API")

try:
    from fastapi import FastAPI, Depends, Header, HTTPException, Request, APIRouter
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.gzip import GZipMiddleware
except ImportError:
    FastAPI = None  # type: ignore

try:
    import jwt  # type: ignore
except ImportError:
    jwt = None  # type: ignore

try:
    import numpy as np  # type: ignore
except ImportError:
    np = None  # type: ignore

try:
    import faiss  # type: ignore
except ImportError:
    faiss = None  # type: ignore


class SpigSettings(BaseSettings):
    app_name: str = "SPIG-Core++ API"
    app_env: str = "dev"

    jwt_secret: str = "CHANGE_ME_SPIG_SUPER_SECRET_KEY"
    jwt_algorithm: str = "HS256"
    jwt_access_minutes: int = 30
    jwt_refresh_days: int = 7

    admin_api_key: Optional[str] = None

    hmac_secret: Optional[str] = None

    max_calls_per_minute: int = 60

    dna_schema_version: str = "1.0.0"

    class Config:
        env_prefix = "SPIG_"
        case_sensitive = False


SETTINGS = SpigSettings()


def _validate_settings(settings: SpigSettings) -> None:
    env = settings.app_env.lower()
    if env != "dev":
        if (
            not settings.jwt_secret
            or settings.jwt_secret == "CHANGE_ME_SPIG_SUPER_SECRET_KEY"
            or len(settings.jwt_secret) < 32
        ):
            raise RuntimeError(
                "In non-dev environments, SPIG_JWT_SECRET must be set to a strong value (>=32 chars)."
            )
        if not settings.admin_api_key or settings.admin_api_key.strip() == "":
            raise RuntimeError(
                "In non-dev environments, SPIG_ADMIN_API_KEY must be set to a non-empty value."
            )


_validate_settings(SETTINGS)

CORRELATION_KEY = "correlation_id"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        data: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        corr = getattr(record, CORRELATION_KEY, None)
        if corr:
            data[CORRELATION_KEY] = corr
        for k, v in getattr(record, "__dict__", {}).items():
            if k not in data and k not in (
                "args",
                "msg",
                "created",
                "msecs",
                "relativeCreated",
                "exc_info",
                "exc_text",
                "stack_info",
            ):
                if not k.startswith("_"):
                    data[k] = v
        return json.dumps(data, ensure_ascii=False)


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("spig_core")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    return logger


logger = setup_logging()
security_logger = logging.getLogger("spig_core.security")
event_logger = logging.getLogger("spig_core.events")
audit_logger = logging.getLogger("spig_core.audit")


class EventBus:
    def __init__(self) -> None:
        self._subscribers: Dict[str, List[Callable[[Dict[str, Any]], None]]] = {}
        self._lock = threading.RLock()

    def subscribe(self, topic: str, handler: Callable[[Dict[str, Any]], None]) -> None:
        with self._lock:
            self._subscribers.setdefault(topic, []).append(handler)

    def publish(self, topic: str, payload: Dict[str, Any]) -> None:
        with self._lock:
            handlers = list(self._subscribers.get(topic, []))
        for h in handlers:
            try:
                h(payload)
            except Exception as exc:
                event_logger.error("Event handler failed for topic=%s: %s", topic, exc)


EVENT_BUS = EventBus()


class Role(str, Enum):
    user = "user"
    auditor = "auditor"
    admin = "admin"


@dataclass
class JwtClaims:
    sub: str
    role: Role
    iat: int
    exp: int
    iss: str
    jti: str
    typ: str


class JwtManager:
    def __init__(self, settings: SpigSettings) -> None:
        self.secret = settings.jwt_secret
        self.algorithm = settings.jwt_algorithm
        self.access_minutes = settings.jwt_access_minutes
        self.refresh_days = settings.jwt_refresh_days
        self.issuer = "SPIG-Core++"
        self._revoked: Dict[str, float] = {}
        self._lock = threading.RLock()

    def _encode(self, payload: Dict[str, Any]) -> str:
        if jwt is None:
            raise RuntimeError("PyJWT غير مثبت، لا يمكن إصدار JWT")
        return jwt.encode(payload, self.secret, algorithm=self.algorithm)  # type: ignore

    def _decode(self, token: str) -> Dict[str, Any]:
        if jwt is None:
            raise RuntimeError("PyJWT غير مثبت، لا يمكن التحقق من JWT")
        return jwt.decode(token, self.secret, algorithms=[self.algorithm])  # type: ignore

    def _now(self) -> int:
        return int(time.time())

    def _new_jti(self) -> str:
        return str(uuid.uuid4())

    def issue_access_refresh(self, user_id: str, role: Role = Role.user) -> Dict[str, str]:
        now = self._now()
        jti_access = self._new_jti()
        jti_refresh = self._new_jti()

        base = {
            "sub": user_id,
            "role": role.value,
            "iat": now,
            "iss": self.issuer,
        }

        access_payload = {
            **base,
            "exp": now + self.access_minutes * 60,
            "jti": jti_access,
            "typ": "access",
        }
        refresh_payload = {
            **base,
            "exp": now + self.refresh_days * 86400,
            "jti": jti_refresh,
            "typ": "refresh",
        }

        with self._lock:
            pass

        return {
            "access_token": self._encode(access_payload),
            "refresh_token": self._encode(refresh_payload),
        }

    def rotate_refresh(self, refresh_token: str) -> Dict[str, str]:
        payload = self._decode(refresh_token)
        if payload.get("typ") != "refresh":
            raise PermissionError("Invalid token type")
        jti = payload.get("jti")
        with self._lock:
            if jti in self._revoked:
                raise PermissionError("Refresh token already used / revoked")
            self._revoked[jti] = time.time()
        user_id = payload.get("sub")
        role = Role(payload.get("role", Role.user.value))
        return self.issue_access_refresh(user_id=user_id, role=role)

    def decode_access(self, token: str) -> JwtClaims:
        payload = self._decode(token)
        if payload.get("typ") != "access":
            raise PermissionError("Invalid token type")
        jti = payload.get("jti")
        with self._lock:
            if jti in self._revoked:
                raise PermissionError("Token revoked")
        return JwtClaims(
            sub=payload["sub"],
            role=Role(payload.get("role", Role.user.value)),
            iat=int(payload["iat"]),
            exp=int(payload["exp"]),
            iss=payload.get("iss", self.issuer),
            jti=jti,
            typ="access",
        )


class RateLimitExceeded(Exception):
    pass


@dataclass
class AuditEvent:
    timestamp: str
    action: str
    actor: str
    meta: Dict[str, Any] = field(default_factory=dict)


class SecurityManager:
    def __init__(self, settings: SpigSettings, jwt_manager: JwtManager) -> None:
        self.settings = settings
        self.jwt_manager = jwt_manager
        self._calls: Dict[str, List[float]] = {}
        self._audit: List[AuditEvent] = []
        self._lock = threading.RLock()

    def _cleanup(self, key: str) -> None:
        now = time.time()
        with self._lock:
            self._calls.setdefault(key, [])
            self._calls[key] = [t for t in self._calls[key] if now - t < 60.0]

    def _register_call(self, key: str) -> None:
        self._cleanup(key)
        with self._lock:
            if len(self._calls[key]) >= self.settings.max_calls_per_minute:
                raise RateLimitExceeded("Rate limit exceeded")
            self._calls[key].append(time.time())

    def verify_hmac(self, raw_body: bytes, signature: Optional[str]) -> None:
        if not self.settings.hmac_secret:
            return
        if not signature:
            raise PermissionError("Missing HMAC signature")
        expected = hmac.new(
            self.settings.hmac_secret.encode("utf-8"),
            msg=raw_body,
            digestmod=hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(expected, signature):
            raise PermissionError("Invalid HMAC signature")

    def audit(self, action: str, actor: str, meta: Optional[Dict[str, Any]] = None) -> None:
        event = AuditEvent(
            timestamp=datetime.utcnow().isoformat(),
            action=action,
            actor=actor,
            meta=meta or {},
        )
        with self._lock:
            self._audit.append(event)
            if len(self._audit) > 10_000:
                self._audit = self._audit[-10_000:]
        audit_logger.info("AUDIT %s %s %s", actor, action, meta or {})

    def get_audit_tail(self, limit: int = 100) -> List[AuditEvent]:
        with self._lock:
            return list(self._audit[-limit:])

    def check_role(self, claims: JwtClaims, required: Role) -> None:
        hierarchy = {Role.user: 1, Role.auditor: 2, Role.admin: 3}
        if hierarchy[claims.role] < hierarchy[required]:
            raise PermissionError("Insufficient role")

    def check_abac(
        self,
        claims: JwtClaims,
        resource: str,
        action: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        context = context or {}
        if resource.startswith("audit") and claims.role not in {Role.admin, Role.auditor}:
            raise PermissionError("ABAC: audit access denied")

    def authenticate_request(
        self,
        authorization: str,
        raw_body: bytes,
        hmac_signature: Optional[str],
        client_id: Optional[str] = None,
        endpoint: Optional[str] = None,
    ) -> JwtClaims:
        if not authorization or not authorization.startswith("Bearer "):
            raise PermissionError("Authorization header must be 'Bearer <token>'")
        token = authorization.split(" ", 1)[1].strip()
        claims = self.jwt_manager.decode_access(token)

        rate_key = claims.sub
        if client_id:
            rate_key = f"{rate_key}:{client_id}"
        if endpoint:
            rate_key = f"{rate_key}:{endpoint}"
        self._register_call(rate_key)

        self.verify_hmac(raw_body, hmac_signature)
        return claims


JWT_MANAGER = JwtManager(SETTINGS)
SECURITY = SecurityManager(SETTINGS, JWT_MANAGER)


class EventType(str, Enum):
    LOGIN = "LOGIN"
    SERVICE_USAGE = "SERVICE_USAGE"
    FRAUD_SUSPECTED = "FRAUD_SUSPECTED"
    OTHER = "OTHER"


class ServiceCategory(str, Enum):
    IDENTITY = "IDENTITY"
    TRAVEL = "TRAVEL"
    RESIDENCY = "RESIDENCY"
    VEHICLE = "VEHICLE"
    SECURITY = "SECURITY"
    OTHER = "OTHER"


class Channel(str, Enum):
    WEB = "WEB"
    MOBILE = "MOBILE"
    CALL_CENTER = "CALL_CENTER"
    IN_PERSON = "IN_PERSON"


@dataclass
class UserIdentitySignals:
    is_citizen: bool = True
    is_resident: bool = False
    is_visitor: bool = False
    device_trust_score: float = 0.7
    biometric_confidence: float = 0.7
    multi_factor_enabled: bool = True


@dataclass
class UserBehaviorSignals:
    last_login_days_ago: int = 1
    total_logins_90d: int = 10
    services_used_90d: int = 15
    distinct_services_90d: int = 5
    avg_session_length_minutes: float = 8.0
    anomalies_count_90d: int = 0


@dataclass
class DocumentStatus:
    name: str
    days_to_expiry: int
    is_critical: bool = True


@dataclass
class UsageEvent:
    timestamp: datetime
    event_type: EventType
    service_category: Optional[ServiceCategory] = None
    channel: Optional[Channel] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UserProfile:
    user_id: str
    identity: UserIdentitySignals
    behavior: UserBehaviorSignals
    documents: List[DocumentStatus] = field(default_factory=list)
    events: List[UsageEvent] = field(default_factory=list)


@dataclass
class EngineFeatures:
    user_id: str
    expiry_risk: float
    engagement_score: float
    inactivity_risk: float
    anomaly_risk: float
    device_trust: float
    biometric_trust: float
    multifactor_bonus: float
    digital_dna_vector: List[float]


@dataclass
class DigitalDNA:
    user_id: str
    version: str
    vector: List[float]

    def normalize(self) -> None:
        if not self.vector:
            return
        vmin = min(self.vector)
        vmax = max(self.vector)
        if vmax == vmin:
            self.vector = [0.5] * len(self.vector)
        else:
            self.vector = [(v - vmin) / (vmax - vmin) for v in self.vector]


class FeatureEngineer:
    def __init__(self) -> None:
        self.max_logins = 40.0
        self.max_services = 30.0
        self.max_distinct = 10.0
        self.max_session = 20.0
        self.max_anomalies = 5.0

    def _expiry_risk(self, docs: List[DocumentStatus]) -> float:
        if not docs:
            return 0.0
        scores = []
        for d in docs:
            if d.days_to_expiry <= 0:
                s = 1.0
            elif d.days_to_expiry <= 30:
                s = 0.85 if d.is_critical else 0.7
            elif d.days_to_expiry <= 90:
                s = 0.5 if d.is_critical else 0.35
            else:
                s = 0.15 if d.is_critical else 0.05
            scores.append(s)
        return sum(scores) / len(scores)

    def _engagement(self, b: UserBehaviorSignals) -> float:
        login_score = min(b.total_logins_90d / self.max_logins, 1.0)
        services_score = min(b.services_used_90d / self.max_services, 1.0)
        diversity_score = min(b.distinct_services_90d / self.max_distinct, 1.0)
        session_score = min(b.avg_session_length_minutes / self.max_session, 1.0)
        return (
            0.25 * login_score
            + 0.35 * services_score
            + 0.2 * diversity_score
            + 0.2 * session_score
        )

    def _inactivity(self, b: UserBehaviorSignals) -> float:
        d = b.last_login_days_ago
        if d <= 1:
            return 0.1
        if d <= 7:
            return 0.25
        if d <= 30:
            return 0.4
        if d <= 90:
            return 0.7
        return 0.9

    def _anomaly(self, b: UserBehaviorSignals, events: List[UsageEvent]) -> float:
        base = min(b.anomalies_count_90d / self.max_anomalies, 1.0)
        if any(e.event_type == EventType.FRAUD_SUSPECTED for e in events):
            base = max(base, 0.7)
        return base

    def _multifactor_bonus(self, identity: UserIdentitySignals) -> float:
        return 0.15 if identity.multi_factor_enabled else 0.0

    def _build_dna(
        self,
        expiry: float,
        engagement: float,
        inactivity: float,
        anomaly: float,
        device: float,
        biometric: float,
        multifactor: float,
        b: UserBehaviorSignals,
    ) -> List[float]:
        vec = [
            expiry,
            engagement,
            inactivity,
            anomaly,
            device,
            biometric,
            multifactor,
            min(b.total_logins_90d / self.max_logins, 1.0),
            min(b.services_used_90d / self.max_services, 1.0),
            min(b.distinct_services_90d / self.max_distinct, 1.0),
            min(b.avg_session_length_minutes / self.max_session, 1.0),
        ]
        return vec

    def build_features(self, profile: UserProfile) -> EngineFeatures:
        expiry = self._expiry_risk(profile.documents)
        engagement = self._engagement(profile.behavior)
        inactivity = self._inactivity(profile.behavior)
        anomaly = self._anomaly(profile.behavior, profile.events)
        device = profile.identity.device_trust_score
        biometric = profile.identity.biometric_confidence
        multifactor = self._multifactor_bonus(profile.identity)
        dna_vec = self._build_dna(
            expiry,
            engagement,
            inactivity,
            anomaly,
            device,
            biometric,
            multifactor,
            profile.behavior,
        )
        return EngineFeatures(
            user_id=profile.user_id,
            expiry_risk=expiry,
            engagement_score=engagement,
            inactivity_risk=inactivity,
            anomaly_risk=anomaly,
            device_trust=device,
            biometric_trust=biometric,
            multifactor_bonus=multifactor,
            digital_dna_vector=dna_vec,
        )


class FeatureStoreProtocol(Protocol):
    def put(self, f: EngineFeatures) -> None:
        ...

    def get(self, user_id: str) -> Optional[EngineFeatures]:
        ...


class InMemoryFeatureStore(FeatureStoreProtocol):
    def __init__(self) -> None:
        self._store: Dict[str, EngineFeatures] = {}
        self._lock = threading.RLock()

    def put(self, f: EngineFeatures) -> None:
        with self._lock:
            self._store[f.user_id] = f

    def get(self, user_id: str) -> Optional[EngineFeatures]:
        with self._lock:
            return self._store.get(user_id)


FEATURE_STORE: FeatureStoreProtocol = InMemoryFeatureStore()


@dataclass
class SimilarityResult:
    reference_id: str
    cosine_similarity: float


class SimilarityEngine:
    def __init__(self) -> None:
        self._dna_index: Dict[str, DigitalDNA] = {}
        self._faiss_index = None
        self._id_to_idx: Dict[str, int] = {}
        self._idx_to_id: Dict[int, str] = {}
        self._lock = threading.RLock()

    def index_dna_bulk(self, dnas: List[DigitalDNA]) -> None:
        with self._lock:
            self._dna_index = {d.user_id: d for d in dnas}
            if faiss is None or np is None or not dnas:
                self._faiss_index = None
                self._id_to_idx = {}
                self._idx_to_id = {}
                return
            dim = len(dnas[0].vector)
            xb = np.array([d.vector for d in dnas], dtype="float32")
            index = faiss.IndexFlatIP(dim)  # type: ignore
            index.add(xb)
            self._faiss_index = index
            self._id_to_idx = {d.user_id: i for i, d in enumerate(dnas)}
            self._idx_to_id = {i: d.user_id for i, d in enumerate(dnas)}

    def upsert_dna(self, dna: DigitalDNA) -> None:
        with self._lock:
            all_dnas = list(self._dna_index.values())
            existing_ids = {d.user_id for d in all_dnas}
            if dna.user_id not in existing_ids:
                all_dnas.append(dna)
            else:
                all_dnas = [d for d in all_dnas if d.user_id != dna.user_id] + [dna]
            self.index_dna_bulk(all_dnas)

    def most_similar(self, dna: DigitalDNA, top_k: int = 3) -> List[SimilarityResult]:
        with self._lock:
            if not self._dna_index:
                return []

            if faiss is None or np is None or self._faiss_index is None:
                res: List[SimilarityResult] = []
                for uid, other in self._dna_index.items():
                    if uid == dna.user_id:
                        continue
                    num = sum(a * b for a, b in zip(dna.vector, other.vector))
                    na = (sum(a * a for a in dna.vector) ** 0.5) or 1e-8
                    nb = (sum(b * b for b in other.vector) ** 0.5) or 1e-8
                    cos = num / (na * nb)
                    res.append(SimilarityResult(reference_id=uid, cosine_similarity=cos))
                res.sort(key=lambda r: r.cosine_similarity, reverse=True)
                return res[:top_k]

            xq = np.array([dna.vector], dtype="float32")  # type: ignore
            D, I = self._faiss_index.search(xq, top_k + 1)  # type: ignore
            results: List[SimilarityResult] = []
            for dist, idx in zip(D[0], I[0]):
                if idx < 0:
                    continue
                uid = self._idx_to_id.get(int(idx))
                if uid is None or uid == dna.user_id:
                    continue
                results.append(
                    SimilarityResult(
                        reference_id=uid,
                        cosine_similarity=float(dist),
                    )
                )
            return results[:top_k]


SIMILARITY_ENGINE = SimilarityEngine()


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


@dataclass
class TrustIndexResult:
    user_id: str
    trust_index: float
    risk_level: RiskLevel
    explanation: str
    recommended_action: str


class TrustEngine:
    def __init__(self, settings: SpigSettings) -> None:
        self.settings = settings
        self.low_threshold = 70.0
        self.medium_threshold = 40.0

    def _rule_score(self, f: EngineFeatures) -> float:
        positive = (
            0.35 * f.engagement_score
            + 0.25 * f.device_trust
            + 0.2 * f.biometric_trust
            + 0.1 * f.multifactor_bonus
        )
        negative = (
            0.4 * f.expiry_risk
            + 0.35 * f.inactivity_risk
            + 0.25 * f.anomaly_risk
        )
        base = 0.5 + (positive - negative)
        return max(0.0, min(1.0, base))

    def compute(self, f: EngineFeatures) -> TrustIndexResult:
        score = self._rule_score(f)
        trust_index = round(score * 100.0, 2)
        if trust_index >= self.low_threshold:
            risk = RiskLevel.LOW
        elif trust_index >= self.medium_threshold:
            risk = RiskLevel.MEDIUM
        else:
            risk = RiskLevel.HIGH

        explanation_parts: List[str] = []
        if f.engagement_score > 0.7:
            explanation_parts.append("تفاعل مرتفع مع المنصة.")
        if f.expiry_risk > 0.7:
            explanation_parts.append("وثائق قريبة من الانتهاء.")
        if f.inactivity_risk > 0.6:
            explanation_parts.append("فترة خمول طويلة منذ آخر دخول.")
        if f.anomaly_risk > 0.6:
            explanation_parts.append("وجود مؤشرات شذوذ سلوكي.")
        if f.multifactor_bonus > 0.0:
            explanation_parts.append("تفعيل التحقق الثنائي يعزّز الثقة.")
        if not explanation_parts:
            explanation_parts.append("سلوك مستقر وبيانات متوازنة.")

        if risk == RiskLevel.LOW:
            rec = "تقديم خدمات استباقية متقدمة وتسهيل التجديدات التلقائية."
        elif risk == RiskLevel.MEDIUM:
            rec = "تعزيز التنبيهات وتجربة استخدام إرشادية مع مراقبة خفيفة."
        else:
            rec = "تشديد ضوابط التحقق وتفعيل مراجعة يدوية للمعاملات الحساسة."

        return TrustIndexResult(
            user_id=f.user_id,
            trust_index=trust_index,
            risk_level=risk,
            explanation=" ".join(explanation_parts),
            recommended_action=rec,
        )


class IDSResult(BaseModel):
    suspicious: bool
    reason: Optional[str] = None


def ids_inspect(features: EngineFeatures, trust: TrustIndexResult) -> IDSResult:
    if trust.risk_level == RiskLevel.HIGH and features.anomaly_risk > 0.6:
        return IDSResult(suspicious=True, reason="High risk + anomaly > 0.6")
    return IDSResult(suspicious=False)


class TransactionType(str, Enum):
    GOV_SERVICE = "GOV_SERVICE"
    PAYMENT = "PAYMENT"
    TRANSFER = "TRANSFER"
    LOGIN = "LOGIN"
    OTHER = "OTHER"


@dataclass
class TransactionEvent:
    tx_id: str
    user_id: str
    tx_type: TransactionType
    amount: float
    timestamp: datetime
    ip: Optional[str] = None
    device_id: Optional[str] = None
    location: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class FraudFeatures:
    user_id: str
    tx_id: str
    amount_norm: float
    velocity_score: float
    geo_change_score: float
    device_switch_score: float
    time_anomaly_score: float
    base_trust_index: float
    composite_vector: List[float]


@dataclass
class FraudScoreResult:
    user_id: str
    tx_id: str
    fraud_score: float
    risk_band: str
    explanation: str


class FraudFeatureEngineer:
    def __init__(self) -> None:
        self.max_amount_ref = 100_000.0

    def _amount_norm(self, amount: float) -> float:
        return min(amount / self.max_amount_ref, 1.0)

    def _velocity_score(self, tx: TransactionEvent, history: List[TransactionEvent]) -> float:
        window = timedelta(minutes=10)
        recent = [h for h in history if tx.timestamp - h.timestamp <= window]
        if len(recent) == 0:
            return 0.1
        if len(recent) == 1:
            return 0.3
        if len(recent) <= 3:
            return 0.6
        return 0.85

    def _geo_change_score(self, tx: TransactionEvent, history: List[TransactionEvent]) -> float:
        if not tx.location:
            return 0.3
        last = next((h for h in reversed(history) if h.location), None)
        if not last:
            return 0.3
        if last.location == tx.location:
            return 0.1
        return 0.8

    def _device_switch_score(self, tx: TransactionEvent, history: List[TransactionEvent]) -> float:
        if not tx.device_id:
            return 0.5
        unique_devices = {h.device_id for h in history if h.device_id} | {tx.device_id}
        if len(unique_devices) == 1:
            return 0.1
        if len(unique_devices) == 2:
            return 0.4
        return 0.8

    def _time_anomaly_score(self, tx: TransactionEvent, history: List[TransactionEvent]) -> float:
        hour = tx.timestamp.hour
        if 2 <= hour <= 4:
            return 0.8
        return 0.2

    def build_features(self, tx: TransactionEvent, history: List[TransactionEvent]) -> FraudFeatures:
        core_features: Optional[EngineFeatures] = FEATURE_STORE.get(tx.user_id)
        base_trust = core_features.engagement_score if core_features else 0.5

        amount_norm = self._amount_norm(tx.amount)
        velocity = self._velocity_score(tx, history)
        geo = self._geo_change_score(tx, history)
        device = self._device_switch_score(tx, history)
        time_anom = self._time_anomaly_score(tx, history)

        composite = [
            amount_norm,
            velocity,
            geo,
            device,
            time_anom,
            base_trust,
        ]

        return FraudFeatures(
            user_id=tx.user_id,
            tx_id=tx.tx_id,
            amount_norm=amount_norm,
            velocity_score=velocity,
            geo_change_score=geo,
            device_switch_score=device,
            time_anomaly_score=time_anom,
            base_trust_index=base_trust,
            composite_vector=composite,
        )


FRAUD_FE = FraudFeatureEngineer()
_FAKE_HISTORY: Dict[str, List[TransactionEvent]] = {}
_FAKE_HISTORY_LOCK = threading.RLock()


@dataclass
class SpigUserResult:
    profile: UserProfile
    features: EngineFeatures
    dna: DigitalDNA
    trust: TrustIndexResult
    ids: IDSResult
    similar: List[SimilarityResult]


class SpigEngine:
    def __init__(self) -> None:
        self.fe = FeatureEngineer()
        self.trust_engine = TrustEngine(SETTINGS)

    def score_profile(self, profile: UserProfile) -> SpigUserResult:
        features = self.fe.build_features(profile)
        FEATURE_STORE.put(features)

        dna = DigitalDNA(
            user_id=profile.user_id,
            version=SETTINGS.dna_schema_version,
            vector=list(features.digital_dna_vector),
        )
        dna.normalize()

        trust = self.trust_engine.compute(features)
        ids_result = ids_inspect(features, trust)

        EVENT_BUS.publish(
            "user.scored",
            {
                "user_id": profile.user_id,
                "trust_index": trust.trust_index,
                "risk_level": trust.risk_level.value,
            },
        )

        SIMILARITY_ENGINE.upsert_dna(dna)
        similar = SIMILARITY_ENGINE.most_similar(dna, top_k=3)

        return SpigUserResult(
            profile=profile,
            features=features,
            dna=dna,
            trust=trust,
            ids=ids_result,
            similar=similar,
        )


ENGINE = SpigEngine()


class DocumentStatusSchema(BaseModel):
    name: str
    days_to_expiry: int
    is_critical: bool = True


class UserIdentitySignalsSchema(BaseModel):
    is_citizen: bool = True
    is_resident: bool = False
    is_visitor: bool = False
    device_trust_score: float = Field(0.7, ge=0.0, le=1.0)
    biometric_confidence: float = Field(0.7, ge=0.0, le=1.0)
    multi_factor_enabled: bool = True


class UserBehaviorSignalsSchema(BaseModel):
    last_login_days_ago: int = Field(1, ge=0)
    total_logins_90d: int = Field(10, ge=0)
    services_used_90d: int = Field(15, ge=0)
    distinct_services_90d: int = Field(5, ge=0)
    avg_session_length_minutes: float = Field(8.0, ge=0.0)
    anomalies_count_90d: int = Field(0, ge=0)


class UsageEventSchema(BaseModel):
    timestamp: datetime
    event_type: EventType
    service_category: Optional[ServiceCategory] = None
    channel: Optional[Channel] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ScoreRequest(BaseModel):
    user_id: str
    identity: UserIdentitySignalsSchema
    behavior: UserBehaviorSignalsSchema
    documents: List[DocumentStatusSchema] = Field(default_factory=list)
    events: List[UsageEventSchema] = Field(default_factory=list)


class SimilarUserSchema(BaseModel):
    reference_id: str
    cosine_similarity: float


class ScoreResponse(BaseModel):
    user_id: str
    trust_index: float
    risk_level: RiskLevel
    explanation: str
    recommended_action: str
    dna_schema_version: str
    ids_suspicious: bool
    ids_reason: Optional[str] = None
    similar_users: List[SimilarUserSchema] = Field(default_factory=list)


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class TokenRequestSchema(BaseModel):
    user_id: str = Field(..., description="هوية المستخدم التي سيتم إصدار التوكين لها")
    role: Role = Field(Role.user, description="دور المستخدم (user / auditor / admin)")


class RefreshRequestSchema(BaseModel):
    refresh_token: str = Field(..., description="Refresh token الحالي المطلوب تدويره")


class AuditEventSchema(BaseModel):
    timestamp: str
    action: str
    actor: str
    meta: Dict[str, Any]


class TransactionEventSchema(BaseModel):
    tx_id: str
    user_id: str
    tx_type: TransactionType = TransactionType.GOV_SERVICE
    amount: float = Field(ge=0.0)
    timestamp: datetime
    ip: Optional[str] = None
    device_id: Optional[str] = None
    location: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class FraudScoreResponse(BaseModel):
    user_id: str
    tx_id: str
    fraud_score: float
    risk_band: str
    explanation: str


app = None

if FastAPI is not None:
    app = FastAPI(
        title=SETTINGS.app_name,
        version="1.0.0",
        description="SPIG-Core++ – Sovereign Predictive Identity Core (Trust + Fraud)",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(GZipMiddleware, minimum_size=1024)

    @app.middleware("http")
    async def add_correlation_id(request: Request, call_next):
        correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
        request.state.correlation_id = correlation_id
        start = time.time()
        try:
            response = await call_next(request)
        except Exception as exc:
            duration = (time.time() - start) * 1000.0
            logger.error(
                "request_error",
                extra={
                    CORRELATION_KEY: correlation_id,
                    "duration_ms": round(duration, 2),
                    "path": str(request.url.path),
                    "method": request.method,
                    "error": str(exc),
                },
            )
            raise
        duration = (time.time() - start) * 1000.0
        response.headers["X-Correlation-ID"] = correlation_id
        logger.info(
            "request",
            extra={
                CORRELATION_KEY: correlation_id,
                "duration_ms": round(duration, 2),
                "path": str(request.url.path),
                "method": request.method,
                "status_code": response.status_code,
            },
        )
        return response

    async def get_current_claims(
        request: Request,
        authorization: str = Header(..., alias="Authorization"),
        hmac_signature: Optional[str] = Header(None, alias="X-SPIG-Signature"),
    ) -> JwtClaims:
        raw_body = await request.body()
        client_ip = request.client.host if request.client else None
        try:
            claims = SECURITY.authenticate_request(
                authorization=authorization,
                raw_body=raw_body,
                hmac_signature=hmac_signature,
                client_id=client_ip,
                endpoint=request.url.path,
            )
            return claims
        except RateLimitExceeded as exc:
            raise HTTPException(status_code=429, detail=str(exc))
        except PermissionError as exc:
            raise HTTPException(status_code=401, detail=str(exc))

    async def verify_admin_api_key(
        x_admin_api_key: Optional[str] = Header(None, alias="X-Admin-API-Key"),
    ) -> bool:
        if SETTINGS.admin_api_key:
            if not x_admin_api_key or x_admin_api_key != SETTINGS.admin_api_key:
                raise HTTPException(status_code=401, detail="invalid admin api key")
        return True

    router_auth = APIRouter(prefix="/api/v1/auth", tags=["auth"])
    router_core = APIRouter(prefix="/api/v1", tags=["score"])
    router_fraud = APIRouter(prefix="/api/v1/fraud", tags=["fraud"])

    @router_auth.post(
        "/token", response_model=TokenResponse, summary="إصدار Access/Refresh Token"
    )
    async def issue_token(
        payload: TokenRequestSchema,
        _: bool = Depends(verify_admin_api_key),
    ) -> TokenResponse:
        user_id = payload.user_id
        role_enum = payload.role
        tokens = JWT_MANAGER.issue_access_refresh(user_id=user_id, role=role_enum)
        SECURITY.audit("issue_token", actor=user_id, meta={"role": role_enum.value})
        return TokenResponse(**tokens)

    @router_auth.post(
        "/refresh", response_model=TokenResponse, summary="تدوير Refresh Token"
    )
    async def refresh_token(payload: RefreshRequestSchema) -> TokenResponse:
        refresh = payload.refresh_token
        if not refresh:
            raise HTTPException(status_code=400, detail="refresh_token required")
        try:
            tokens = JWT_MANAGER.rotate_refresh(refresh)
            if jwt is None:
                raise HTTPException(
                    status_code=500, detail="JWT library not available"
                )
            sub = jwt.decode(
                refresh,
                SETTINGS.jwt_secret,
                algorithms=[SETTINGS.jwt_algorithm],
            ).get("sub")  # type: ignore
            SECURITY.audit("refresh_token", actor=sub)
            return TokenResponse(**tokens)
        except PermissionError as exc:
            raise HTTPException(status_code=401, detail=str(exc))
        except Exception:
            raise HTTPException(status_code=401, detail="invalid refresh token")

    @router_core.post(
        "/score",
        response_model=ScoreResponse,
        summary="حساب مؤشر الثقة Trust Index",
    )
    async def score_endpoint(
        req: ScoreRequest,
        claims: JwtClaims = Depends(get_current_claims),
    ) -> ScoreResponse:
        try:
            SECURITY.check_abac(claims, resource="score", action="invoke")
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc))

        profile = UserProfile(
            user_id=req.user_id,
            identity=UserIdentitySignals(**req.identity.dict()),
            behavior=UserBehaviorSignals(**req.behavior.dict()),
            documents=[DocumentStatus(**d.dict()) for d in req.documents],
            events=[
                UsageEvent(
                    timestamp=e.timestamp,
                    event_type=e.event_type,
                    service_category=e.service_category,
                    channel=e.channel,
                    metadata=e.metadata,
                )
                for e in req.events
            ],
        )
        result = ENGINE.score_profile(profile)
        SECURITY.audit(
            "score_user",
            actor=claims.sub,
            meta={
                "payload_user_id": req.user_id,
                "risk_level": result.trust.risk_level.value,
            },
        )

        return ScoreResponse(
            user_id=result.profile.user_id,
            trust_index=result.trust.trust_index,
            risk_level=result.trust.risk_level,
            explanation=result.trust.explanation,
            recommended_action=result.trust.recommended_action,
            dna_schema_version=result.dna.version,
            ids_suspicious=result.ids.suspicious,
            ids_reason=result.ids.reason,
            similar_users=[
                SimilarUserSchema(
                    reference_id=s.reference_id,
                    cosine_similarity=s.cosine_similarity,
                )
                for s in result.similar
            ],
        )

    @router_fraud.post(
        "/score",
        response_model=FraudScoreResponse,
        summary="حساب مؤشر الاحتيال Fraud Score",
    )
    async def fraud_score_endpoint(
        payload: TransactionEventSchema,
        claims: JwtClaims = Depends(get_current_claims),
    ) -> FraudScoreResponse:
        tx = TransactionEvent(
            tx_id=payload.tx_id,
            user_id=payload.user_id,
            tx_type=payload.tx_type,
            amount=payload.amount,
            timestamp=payload.timestamp,
            ip=payload.ip,
            device_id=payload.device_id,
            location=payload.location,
            metadata=payload.metadata or {},
        )

        with _FAKE_HISTORY_LOCK:
            history = _FAKE_HISTORY.get(tx.user_id, []).copy()
            feats = FRAUD_FE.build_features(tx, history)

            risk_raw = (
                0.3 * feats.velocity_score
                + 0.25 * feats.geo_change_score
                + 0.2 * feats.device_switch_score
                + 0.15 * feats.time_anomaly_score
                + 0.1 * (1.0 - feats.base_trust_index)
            )
            risk_raw = max(0.0, min(1.0, risk_raw))
            fraud_score = round(risk_raw * 100.0, 2)

            if fraud_score < 30:
                band = "LOW"
                expl = "نمط المعاملة مستقر، مع مؤشرات منخفضة للاشتباه."
            elif fraud_score < 60:
                band = "MEDIUM"
                expl = "هناك بعض الإشارات التي تستدعي مراقبة إضافية دون حظر فوري."
            elif fraud_score < 85:
                band = "HIGH"
                expl = "وجود مؤشرات قوية تستدعي تفعيل ضوابط تحقق إضافية."
            else:
                band = "CRITICAL"
                expl = "احتمال احتيال مرتفع جدًا، يفضل إيقاف العملية والتحويل للمراجعة اليدوية."

            _FAKE_HISTORY.setdefault(tx.user_id, []).append(tx)

        SECURITY.audit(
            "fraud_score",
            actor=claims.sub,
            meta={
                "tx_id": tx.tx_id,
                "user_id": tx.user_id,
                "fraud_score": fraud_score,
                "risk_band": band,
            },
        )

        return FraudScoreResponse(
            user_id=tx.user_id,
            tx_id=tx.tx_id,
            fraud_score=fraud_score,
            risk_band=band,
            explanation=expl,
        )

    @app.get("/api/v1/health", summary="Health Check")
    async def health() -> Dict[str, Any]:
        return {
            "status": "ok",
            "app": SETTINGS.app_name,
            "env": SETTINGS.app_env,
            "dna_schema": SETTINGS.dna_schema_version,
        }

    @app.get(
        "/api/v1/audit",
        response_model=List[AuditEventSchema],
        summary="عرض آخر أحداث التدقيق (Auditor/Admin)",
    )
    async def get_audit(
        limit: int = 50,
        claims: JwtClaims = Depends(get_current_claims),
    ) -> List[AuditEventSchema]:
        try:
            SECURITY.check_role(claims, required=Role.auditor)
            SECURITY.check_abac(claims, resource="audit", action="read")
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc))
        events = SECURITY.get_audit_tail(limit=limit)
        return [AuditEventSchema(**asdict(e)) for e in events]

    app.include_router(router_auth)
    app.include_router(router_core)
    app.include_router(router_fraud)


def _demo_build_profile(user_id: str = "demo_user") -> UserProfile:
    identity = UserIdentitySignals()
    behavior = UserBehaviorSignals()
    documents = [
        DocumentStatus(name="National ID", days_to_expiry=180, is_critical=True),
        DocumentStatus(name="Passport", days_to_expiry=30, is_critical=True),
    ]
    now = datetime.utcnow()
    events = [
        UsageEvent(timestamp=now, event_type=EventType.LOGIN, channel=Channel.MOBILE),
        UsageEvent(
            timestamp=now,
            event_type=EventType.SERVICE_USAGE,
            service_category=ServiceCategory.IDENTITY,
            channel=Channel.MOBILE,
        ),
    ]
    return UserProfile(
        user_id=user_id,
        identity=identity,
        behavior=behavior,
        documents=documents,
        events=events,
    )


if __name__ == "__main__":
    profile = _demo_build_profile()
    result = ENGINE.score_profile(profile)
    print("User:", result.profile.user_id)
    print("Trust Index:", result.trust.trust_index, result.trust.risk_level.value)
    print("Explanation:", result.trust.explanation)
    print("Recommended Action:", result.trust.recommended_action)
    print("Digital DNA:", result.dna.vector)
    print("IDS Suspicious:", result.ids.suspicious, "-", result.ids.reason)
    print(
        "Similar Users:",
        [f"{s.reference_id}:{s.cosine_similarity:.3f}" for s in result.similar],
    )
