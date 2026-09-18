import numpy as np
import math
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from src.preprocessor import TextPreprocessor

BASE_MAP = {
    'A': {
        'shall', 'must', 'will', 'provide', 'enable', 'support', 'allow', 'implement',
        'execute', 'process', 'generate', 'display', 'store', 'retrieve', 'transmit',
        'receive', 'validate', 'verify', 'calculate', 'compute', 'update', 'delete',
        'create', 'modify', 'send', 'record', 'maintain', 'manage', 'handle', 'schedule',
        'allocate', 'deliver', 'notify', 'trigger', 'initiate', 'require', 'specify',
        'output', 'input', 'produce', 'export', 'import', 'accept', 'check', 'convert',
        'format', 'track', 'search', 'print', 'log', 'render', 'filter', 'sort'
    },
    'T': {
        'user', 'interface', 'usable', 'easy', 'simple', 'intuitive', 'friendly',
        'accessible', 'screen', 'button', 'menu', 'dialog', 'form', 'report',
        'dashboard', 'visualization', 'navigation', 'help', 'documentation', 'training',
        'look', 'feel', 'color', 'font', 'theme', 'style', 'layout', 'readable',
        'visual', 'guideline', 'clear', 'icon', 'graphic', 'presentation', 'aesthetic',
        'appearance', 'view', 'wizard', 'feedback', 'prompt', 'ergonomic', 'understandable',
        'legible', 'viewer', 'viewing', 'distance', 'human'
    },
    'C': {
        'secure', 'security', 'authentication', 'authorization', 'encryption', 'privacy',
        'confidential', 'confidentiality', 'integrity', 'audit', 'compliance', 'credential',
        'certificate', 'firewall', 'token', 'session', 'vulnerability', 'threat', 'attack',
        'access', 'permission', 'role', 'protect', 'protection', 'policy', 'legal',
        'regulation', 'statutory', 'gdpr', 'hipaa', 'license', 'copyright', 'breach',
        'tamper', 'safeguard', 'password', 'login', 'logout', 'sign-in', 'restricted',
        'safe', 'safely', 'hazard', 'standards', 'w3c', 'regulations'
    },
    'G': {
        'fast', 'speed', 'response', 'performance', 'time', 'latency', 'throughput',
        'bandwidth', 'capacity', 'scalability', 'efficiency', 'real-time', 'concurrent',
        'parallel', 'millisecond', 'second', 'minute', 'hour', 'load', 'stress',
        'benchmark', 'optimize', 'cache', 'buffer', 'queue', 'rate', 'frequency',
        'memory', 'cpu', 'overhead', 'available', 'availability', 'uptime', 'downtime',
        'reliable', 'reliability', 'fault-tolerant', 'backup', 'recovery', 'redundancy',
        'failover', 'resilient', 'robust', 'mtbf', 'disaster', 'crash', 'platform',
        'operating', 'system', 'linux', 'windows', 'browser', 'hardware', 'server',
        'simultaneous', 'scaleable', 'simultaneously', 'offline'
    }
}

WORD_TO_BASE = {}
for b, words in BASE_MAP.items():
    for w in words:
        WORD_TO_BASE[w] = b

MONOMERS = ['A', 'T', 'C', 'G']
DIMERS = [b1 + b2 for b1 in MONOMERS for b2 in MONOMERS]

class TrueGenomicFeatureExtractor:
    def __init__(self, max_tfidf_features=50, sbert_model_name='all-MiniLM-L6-v2'):
        self.sbert_model = SentenceTransformer(sbert_model_name)
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_tfidf_features,
            stop_words='english',
            ngram_range=(1, 2),
            sublinear_tf=True
        )

    def text_to_genomic_features(self, text: str) -> list:
        tokens = re.findall(r'\b[a-z0-9\-_]{2,}\b', str(text).lower())
        seq = []
        for t in tokens:
            if t in WORD_TO_BASE:
                seq.append(WORD_TO_BASE[t])
            elif re.match(r'^\d+$', t):
                seq.append('G')
        if not seq:
            seq = ['A']
            
        dna_str = "".join(seq)
        L = len(dna_str)
        c1 = Counter(dna_str)
        
        # 1-mers (4)
        k1_feats = [c1[m] / L for m in MONOMERS]
        
        # 2-mers (16)
        c2 = Counter([dna_str[i:i+2] for i in range(L - 1)])
        L2 = max(1, L - 1)
        k2_feats = [c2[d] / L2 for d in DIMERS]
        
        # Biochemical metrics (3)
        gc_content = (c1['G'] + c1['C']) / L
        at_gc_ratio = (c1['A'] + c1['T']) / (c1['G'] + c1['C'] + 1e-5)
        seq_len_norm = math.log1p(L)
        
        return k1_feats + k2_feats + [gc_content, at_gc_ratio, seq_len_norm]

    def fit(self, texts: list):
        cleaned = [TextPreprocessor.clean_text(t) for t in texts]
        self.tfidf_vectorizer.fit(cleaned)
        return self

    def transform(self, texts: list) -> np.ndarray:
        cleaned = [TextPreprocessor.clean_text(t) for t in texts]
        
        # 1. Biological Genomic Features (23 dimensions)
        genomic_feats = np.array([self.text_to_genomic_features(t) for t in texts], dtype=np.float32)
        
        # 2. TF-IDF (50 dimensions)
        tfidf_feats = self.tfidf_vectorizer.transform(cleaned).toarray()
        
        # 3. SBERT (384 dimensions)
        sbert_feats = self.sbert_model.encode(cleaned, show_progress_bar=False)
        
        # Multimodal Fused Matrix: emphasize biological genomic features (2.0x weight)
        X_fused = np.hstack((genomic_feats * 2.0, tfidf_feats, sbert_feats * 1.5))
        return X_fused

    def fit_transform(self, texts: list) -> np.ndarray:
        self.fit(texts)
        return self.transform(texts)
