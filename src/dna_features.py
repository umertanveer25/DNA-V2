import numpy as np
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from src.preprocessor import TextPreprocessor

DNA_VOCAB = {
    'A': {
        'shall', 'must', 'will', 'provide', 'enable', 'support', 'allow', 'implement',
        'execute', 'process', 'generate', 'display', 'store', 'retrieve', 'transmit',
        'receive', 'validate', 'verify', 'calculate', 'compute', 'update', 'delete',
        'create', 'modify', 'send', 'record', 'maintain', 'manage', 'handle', 'schedule',
        'allocate', 'deliver', 'notify', 'trigger', 'initiate', 'require', 'specify',
        'output', 'input', 'produce', 'export', 'import', 'accept', 'check'
    },
    'T': {
        'user', 'interface', 'usable', 'easy', 'simple', 'intuitive', 'friendly',
        'accessible', 'screen', 'button', 'menu', 'dialog', 'form', 'report',
        'dashboard', 'visualization', 'navigation', 'help', 'documentation', 'training',
        'look', 'feel', 'color', 'font', 'theme', 'style', 'layout', 'readable',
        'visual', 'guideline', 'clear', 'icon', 'graphic', 'presentation', 'aesthetic',
        'appearance', 'view', 'wizard', 'feedback', 'message', 'prompt', 'error-message'
    },
    'C': {
        'secure', 'security', 'authentication', 'authorization', 'encryption', 'privacy',
        'confidential', 'confidentiality', 'integrity', 'audit', 'compliance', 'credential',
        'certificate', 'firewall', 'token', 'session', 'vulnerability', 'threat', 'attack',
        'access', 'permission', 'role', 'protect', 'protection', 'policy', 'legal',
        'regulation', 'statutory', 'gdpr', 'hipaa', 'license', 'copyright', 'breach',
        'tamper', 'safeguard', 'password', 'login', 'logout', 'sign-in', 'restricted'
    },
    'G': {
        'fast', 'speed', 'response', 'performance', 'time', 'latency', 'throughput',
        'bandwidth', 'capacity', 'scalability', 'efficiency', 'real-time', 'concurrent',
        'parallel', 'millisecond', 'second', 'minute', 'hour', 'load', 'stress',
        'benchmark', 'optimize', 'cache', 'buffer', 'queue', 'rate', 'frequency',
        'memory', 'cpu', 'overhead', 'available', 'availability', 'uptime', 'downtime',
        'reliable', 'reliability', 'fault-tolerant', 'backup', 'recovery', 'redundancy',
        'failover', 'resilient', 'robust', 'mtbf', 'disaster', 'crash', 'platform',
        'operating', 'system', 'linux', 'windows', 'browser', 'hardware', 'server'
    }
}

class EnhancedDNAFeatureExtractor:
    def __init__(self, max_tfidf_features=50, sbert_model_name='all-MiniLM-L6-v2'):
        self.sbert_model = SentenceTransformer(sbert_model_name)
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_tfidf_features,
            stop_words='english',
            ngram_range=(1, 2),
            sublinear_tf=True
        )

    def extract_dna_structural_features(self, texts: list) -> np.ndarray:
        dna_features = []
        for text in texts:
            tokens = set(TextPreprocessor.tokenize(text))
            
            # 1. Monomer base counts
            cA = len(tokens.intersection(DNA_VOCAB['A']))
            cT = len(tokens.intersection(DNA_VOCAB['T']))
            cC = len(tokens.intersection(DNA_VOCAB['C']))
            cG = len(tokens.intersection(DNA_VOCAB['G']))
            
            # 2. Dinucleotide cross-coupling interactions
            pAT = cA * cT
            pAG = cA * cG
            pAC = cA * cC
            pTG = cT * cG
            pTC = cT * cC
            pGC = cG * cC
            
            # 3. Biochemical & Information Ratios
            at_ratio = cA / (cA + cT + 1.0)
            gc_ratio = cG / (cG + cC + 1.0)
            atgc_ratio = (cA + cT) / (cG + cC + 1.0)
            
            # 4. Nucleotide Shannon Entropy
            total_bases = cA + cT + cC + cG
            entropy = 0.0
            if total_bases > 0:
                for count in [cA, cT, cC, cG]:
                    if count > 0:
                        p = count / total_bases
                        entropy -= p * math.log2(p)
                        
            vec = [
                cA, cT, cC, cG,
                pAT, pAG, pAC, pTG, pTC, pGC,
                at_ratio, gc_ratio, atgc_ratio, entropy
            ]
            dna_features.append(vec)
            
        return np.array(dna_features, dtype=np.float32)

    def fit(self, texts: list):
        cleaned_texts = [TextPreprocessor.clean_text(t) for t in texts]
        self.tfidf_vectorizer.fit(cleaned_texts)
        return self

    def transform(self, texts: list) -> np.ndarray:
        cleaned_texts = [TextPreprocessor.clean_text(t) for t in texts]
        
        # 1. Structural DNA & Dinucleotide features (14-dim)
        dna_feats = self.extract_dna_structural_features(texts)
        
        # 2. TF-IDF features (50-dim)
        tfidf_feats = self.tfidf_vectorizer.transform(cleaned_texts).toarray()
        
        # 3. SBERT Embeddings (384-dim)
        sbert_feats = self.sbert_model.encode(cleaned_texts, show_progress_bar=False)
        
        # Multimodal fusion representation
        X_fused = np.hstack((dna_feats * 2.0, tfidf_feats, sbert_feats * 1.5))
        return X_fused

    def fit_transform(self, texts: list) -> np.ndarray:
        self.fit(texts)
        return self.transform(texts)
