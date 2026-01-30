"""
Sentence Pattern Extraction Module

Extracts and manages common academic sentence patterns using spaCy NLP.
"""

import os
import json
import re
from typing import List, Dict, Tuple
from collections import defaultdict

# Lazy load spaCy to avoid slow startup
_nlp = None

def get_nlp():
    """Lazy load spaCy model."""
    global _nlp
    if _nlp is None:
        try:
            import spacy
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("spaCy model not found. Run: python -m spacy download en_core_web_sm")
            return None
    return _nlp


# Common academic phrase patterns to look for (100+ patterns)
ACADEMIC_PATTERNS = [
    # ============= ARGUMENTATION & LOGIC =============
    r"(we (show|demonstrate|prove|establish|argue|claim|conjecture|hypothesize) that)",
    r"(it (follows|is clear|can be shown|is evident|is straightforward|is easy to see|is immediate|turns out) that)",
    r"((this|the above|these results?) (implies?|suggests?|indicates?|shows?|reveals?|establishes?) that)",
    r"((implies|suggests|indicates) that)", # Catch standalone occurrences too
    r"(consistent with (the hypothesis|our results|the prediction))",
    r"(support (for|of) the (hypothesis|theory|claim|view))",
    r"(reasoning (suggests|implies) that)",
    r"(in (support|favor) of)",
    r"(argue (that|for|against))",
    r"(evidence (suggests|indicates|shows|supports))",
    r"(in contrast|by contrast|on the other hand|conversely|alternatively)",
    r"(moreover|furthermore|in addition|additionally|also|equally important)",
    r"(however|nevertheless|nonetheless|yet|but|still|even so)",
    r"(therefore|thus|hence|consequently|as a result|it follows that|accordingly)",
    r"(intuitively|roughly speaking|loosely speaking|to see why|the intuition is)",
    r"(observe that|note that|notice that|recall that|importantly)",
    r"(the (key|main|crucial|central|critical|essential) (insight|observation|point|idea) is)",
    r"(as (we|one) (will|shall) see|as (shown|discussed|argued) (below|above|later))",
    
    # ============= STRUCTURE & ROADMAP =============
    r"(in this (paper|section|chapter|note|article)|the rest of (the paper|this section))",
    r"(the (main|key|central|primary) (result|contribution|finding|insight|theorem))",
    r"(we (first|then|next|finally|now|begin by|proceed to) (consider|examine|analyze|turn to|discuss|study))",
    r"(the (following|next|subsequent) (proposition|theorem|lemma|corollary|result|claim))",
    r"(section \\d+ (discusses|presents|analyzes|introduces|develops|contains))",
    r"(the paper is organized as follows|the remainder of the paper|we organize the paper)",
    r"(before proceeding|before (we|turning to)|to (set up|fix ideas|establish notation))",
    r"(we (now|next) (present|state|derive|prove)|turning (now )?to|we are ready to)",
    
    # ============= DEFINITIONS & NOTATION =============
    r"(let|define|denote|suppose|assume|consider) [\w\s]+ (be|as|to be)",
    r"(we (say|call) that|is (said|called) to be|is defined (as|to be))",
    r"(without loss of generality|w\.?l\.?o\.?g\.?)",
    r"(formally|more precisely|to be precise|specifically|in particular)",
    r"(we (write|use|adopt) (the )?notation|by (abuse of|slight abuse of) notation)",
    r"(the set of all|the space of|the collection of|the class of)",
    r"(for (any|all|each|every|some) \w+|for \w+ (sufficiently|large|small))",
    r"(where \w+ (denotes|represents|is|stands for))",
    
    # ============= RESULTS & FINDINGS =============
    r"(the (optimal|equilibrium|efficient|unique|first-best|second-best) (mechanism|allocation|outcome|contract|policy))",
    r"(there exists (a|an|the)?\s?(unique|optimal|efficient|equilibrium)?)",
    r"(if and only if|necessary and sufficient|iff)",
    r"(in (a|the|any) (Nash|Bayesian|Perfect|Sequential|Subgame|Markov) equilibrium)",
    r"(characterizes|characterizing|the characterization|fully characterizes|completely characterizes)",
    r"(is (strictly|weakly)? (increasing|decreasing|monotonic|convex|concave) in)",
    r"(the (unique|only) (equilibrium|solution|mechanism|allocation))",
    r"(admits (a|an) (unique|efficient|closed-form) (solution|characterization))",
    
    # ============= ECONOMICS / GAME THEORY =============
    r"(incentive (compatible|compatibility|constraints?)|IC (constraint)?)",
    r"(individual(ly)? rational(ity)?|participation constraint|IR (constraint)?)",
    r"(first|second) order (condition|stochastic dominance|approach)",
    r"(utility|payoff|profit|welfare|surplus) (function|maximization|maximizing)",
    r"(single( |-)crossing|monotone likelihood ratio|MLRP|assortative matching)",
    r"(information (structure|design|asymmetry|rent)|private information|private type)",
    r"(mechanism (design|designer|problem)|principal(-| )agent|moral hazard|adverse selection)",
    r"(revelation principle|direct mechanism|incentive (feasible|feasibility))",
    r"(implementable|implementation|implements the|implement(s|ing) the)",
    r"(social (welfare|planner|choice)|Pareto (efficient|optimal|improvement|frontier))",
    r"(ex (ante|interim|post)|before|after) (realization|learning|observing)",
    r"(common (knowledge|prior|belief)|public (information|signal|belief))",
    r"(cheap talk|signaling|screening|communication|disclosure)",
    r"(virtual (utility|surplus|valuation)|information rent|rent extraction)",
    r"(envelope (theorem|condition|formula)|marginal (type|agent))",
    r"(ironing|bunching|pooling|separating|semi-separating)",
    r"(bang-bang|corner solution|interior solution|binding constraint)",
    
    # ============= INFORMATION DESIGN & PERSUASION =============
    r"(Bayesian persuasion|information design|signal(ing|s)?)",
    r"(sender|receiver|designer|principal|agent|informed( party)?|uninformed( party)?)",
    r"(posterior (belief|distribution|expectation)|prior (belief|distribution))",
    r"(concavification|concave closure|convexification)",
    r"(obedience (constraint|condition)|incentive (of|for) (the )?(receiver|agent))",
    r"(full (disclosure|revelation|information)|no (disclosure|information))",
    r"(partial (disclosure|revelation|pooling)|information (partition|structure))",
    r"(value of information|informational (content|value)|Blackwell (ordering|informativeness))",
    r"(commitment (power|assumption)|credible|credibility)",
    r"(persuasion (problem|setting|game)|sender's (problem|objective|payoff))",
    r"(receiver's (action|decision|best response)|optimal (action|decision))",
    r"(signal (realization|structure|space)|experiment|test)",
    r"(splitting|garbling|mean-preserving spread|Bayes plausible|martingale)",
    
    # ============= CONTRACT THEORY =============
    r"(moral hazard|hidden (action|effort|information)|unobservable (effort|action))",
    r"(adverse selection|hidden (type|characteristic)|private (type|information))",
    r"(first-best|second-best|third-best|constrained (efficient|optimal))",
    r"(limited liability|wealth constraint|budget constraint)",
    r"(risk (averse|neutral|loving)|risk (sharing|allocation))",
    r"(complete (contract|contracting)|incomplete (contract|contracting))",
    r"(hold-up (problem)?|renegotiation|commitment (problem)?)",
    r"(screening (contract|mechanism)|menu of contracts|nonlinear (pricing|tariff))",
    r"(observ(able|ability)|verifi(able|ability)|contractible)",
    
    # ============= PROBABILITY & BELIEFS =============
    r"(with probability (one|zero|\d|\w+)|almost surely|a\.s\.)",
    r"(in expectation|expected (value|utility|payoff)|E\[)",
    r"(conditional (on|expectation|probability|distribution))",
    r"(Bayes('|'s)? (rule|theorem|formula|updating)|Bayesian (updating|inference))",
    r"(belief (updating|formation|consistency)|consistent beliefs)",
    r"(distribution (over|of)|distributed (according to|as))",
    r"(random (variable|vector|process)|stochastic)",
    r"(independence|independent (of|and identically)|i\.i\.d\.)",
    
    # ============= COMPARISON & BENCHMARK =============
    r"(compared (to|with)|relative to|in comparison (with|to))",
    r"(outperforms|dominates( strictly)?|is (superior|inferior) to)",
    r"(benchmark (case|model)|baseline|standard model)",
    r"(special case|limiting case|as a (limiting|special) case)",
    r"(generalizes|extends|nests|subsumes|encompasses)",
    r"(reduces to|collapses to|coincides with|is equivalent to)",
    
    # ============= LITERATURE & MOTIVATION =============
    r"(little is known about|remains an open question|less attention has been paid)",
    r"(fills this gap|addresses this (issue|gap)|bridges the gap)",
    r"(differs from|unlike|contrary to|in contrast with) [\w\s]+",
    r"(contributes to the literature|builds on|extends (the model of|the work of|the results of))",
    r"(related (literature|work|papers)|the literature on|a growing (literature|body of work))",
    r"(pioneered by|following|in the spirit of|builds upon)",
    r"(has (received|attracted) (considerable|much|significant) attention|well-studied|extensively studied)",
    r"(new (to|in) the literature|first to (show|characterize|study))",
    
    # ============= METHODOLOGY =============
    r"(we (employ|adopt|utilize|use|apply|develop) (a|the|an) (model|framework|approach|technique|method))",
    r"(our (model|framework|setup|approach) (captures|allows for|incorporates|features|accommodates))",
    r"(solving|maximizing|minimizing) the (problem|program|objective|optimization)",
    r"(proof (is in|provided in|appears in|relegated to) (the )?appendix)",
    r"(taking (the )?derivative|first-order (condition|derivative)|FOC)",
    r"(by (backward )?induction|using (standard |straightforward )?arguments)",
    r"(closed-form (solution|expression|characterization)|analytical solution)",
    r"(sufficient (conditions?|statistics?)|necessary (conditions?|requirements?))",
    
    # ============= LIMITATIONS & FUTURE =============
    r"(limitations of|restricted to|focuses only on|confines attention to)",
    r"(leave (for|to) future (research|work)|promising (direction|avenue)|future (work|research))",
    r"(beyond the scope of|does not (address|cover|include|consider))",
    r"(abstrac(t|ting) (from|away)|for (simplicity|tractability)|simplifying assumption)",
    r"(natural (extension|generalization|direction)|interesting (extension|question))",
    r"(robustness (check|analysis)|sensitivity (analysis|to))",
    
    # ============= MATHEMATICAL STATEMENTS =============
    r"(for (all|any|each|every) \w+ (such that|satisfying|with))",
    r"(there (exists|exist) (a|an)?\s?\w+\s?(such that)?)",
    r"(is (bounded|finite|countable|measurable|continuous|differentiable))",
    r"(converges (to|in)|convergence|limiting (behavior|distribution))",
    r"((strictly|weakly) (positive|negative|greater|less|increasing|decreasing))",
    r"(supremum|infimum|maximum|minimum|argmax|argmin)",
]


def extract_sentences(text: str) -> List[str]:
    """
    Extract sentences from text, filtering for quality.
    
    Args:
        text: Raw text from a paper
    
    Returns:
        List of clean sentences
    """
    nlp = get_nlp()
    if nlp is None:
        # Fallback to simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 20]
    
    doc = nlp(text)
    sentences = []
    
    for sent in doc.sents:
        sent_text = sent.text.strip()
        # Filter criteria
        word_count = len(sent_text.split())
        if 5 <= word_count <= 50:  # Reasonable sentence length
            if not re.match(r'^[\d\.\-\(\)]+$', sent_text):  # Not just numbers
                if not sent_text.startswith('http'):  # Not URLs
                    sentences.append(sent_text)
    
    return sentences


def find_pattern_matches(text: str) -> List[Dict]:
    """
    Find academic patterns in text.
    
    Args:
        text: Text to search
    
    Returns:
        List of matches with pattern and context
    """
    matches = []
    sentences = extract_sentences(text)
    
    for sentence in sentences:
        sentence_lower = sentence.lower()
        for pattern in ACADEMIC_PATTERNS:
            match = re.search(pattern, sentence_lower)
            if match:
                matches.append({
                    'pattern': pattern,
                    'matched_text': match.group(0),
                    'sentence': sentence,
                    'category': categorize_pattern(pattern)
                })
    
    return matches


def categorize_pattern(pattern: str) -> str:
    """Categorize a pattern into a semantic group."""
    # Econ/Game Theory - Check FIRST to avoid generic classification
    if any(word in pattern for word in ['incentive', 'rationality', 'condition', 'utility', 'payoff', 
                                      'profit', 'crossing', 'monotone', 'matching', 'information', 
                                      'mechanism', 'principal', 'agent', 'moral', 'adverse', 'risk',
                                      'belief', 'posterior', 'prior', 'strategy', 'nash', 'bayes', 
                                      'equilibrium', 'allocation', 'surplus', 'welfare']):
        return 'Econ/Game Theory'
    elif any(word in pattern for word in ['show', 'demonstrate', 'prove', 'argue', 'claim', 'establish', 'propose', 'suggest', 'implies', 'indicates']):
        return 'Argumentation'
    elif any(word in pattern for word in ['contrast', 'however', 'nevertheless', 'yet', 'conversely', 'unlike', 'differs']):
        return 'Contrast'
    elif any(word in pattern for word in ['therefore', 'thus', 'hence', 'consequently', 'result', 'follows']):
        return 'Conclusion'
    elif any(word in pattern for word in ['moreover', 'furthermore', 'addition']):
        return 'Addition'
    elif any(word in pattern for word in ['let', 'define', 'denote', 'suppose', 'assume', 'consider']):
        return 'Definition/Setup'
    elif any(word in pattern for word in ['paper', 'section', 'contribution', 'finding', 'organized']):
        return 'Structure'
    elif any(word in pattern for word in ['optimal', 'exists', 'unique', 'efficient', 'necessary', 'outcome', 'solution']):
        # 'equilibrium' moved to Econ
        return 'Results'
    elif any(word in pattern for word in ['gap', 'question', 'attention', 'fills', 'contributes', 'extends']):
        return 'Motivation/Gap'
    elif any(word in pattern for word in ['employ', 'adopt', 'utilize', 'model', 'framework', 'solving', 'proof', 'appendix']):
        return 'Methodology'
    elif any(word in pattern for word in ['limitations', 'scope', 'future', 'extensions']):
        return 'Limitations/Future'
    else:
        return 'General Academic'


def build_pattern_library(data_dir: str = None) -> Dict:
    """
    Scan all papers and build a pattern library using fast regex matching.
    
    Args:
        data_dir: Path to data directory (default: ../data relative to this file)
    
    Returns:
        Pattern library dictionary
    """
    from src import database
    
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    
    library_path = os.path.join(data_dir, "pattern_library.json")
    
    # Get all documents from Database
    result = database.get_all_chunks()
    
    # Mock papers list for count
    papers = set()
    for meta in result.get('metadatas', []):
        if meta:
            papers.add(meta.get('title'))
    
    if not result.get('documents'):
        return {"patterns": {}, "total_papers": 0}
    
    pattern_library = defaultdict(lambda: {"count": 0, "examples": [], "category": ""})
    
    # Compile all patterns for speed
    compiled_patterns = []
    for pattern in ACADEMIC_PATTERNS:
        try:
            compiled_patterns.append((pattern, re.compile(pattern, re.IGNORECASE)))
        except Exception:
            pass  # Skip invalid patterns
    
    print(f"Analyzing {len(result.get('documents', []))} chunks with {len(compiled_patterns)} patterns...")
    
    for i, doc in enumerate(result.get('documents', [])):
        if not doc:
            continue
        
        # Fast sentence splitting (no spaCy)
        sentences = _fast_sentence_split(doc)
        
        # Regex Pattern Matching
        for sentence in sentences:
            for pattern_str, pattern_re in compiled_patterns:
                match = pattern_re.search(sentence)
                if match:
                    pattern_library[pattern_str]['count'] += 1
                    pattern_library[pattern_str]['category'] = categorize_pattern(pattern_str)
                    
                    # Store up to 5 examples per pattern
                    if len(pattern_library[pattern_str]['examples']) < 5:
                        source = result['metadatas'][i].get('title', 'Unknown') if result.get('metadatas') else 'Unknown'
                        example = {'sentence': sentence[:200], 'source': source}
                        if example not in pattern_library[pattern_str]['examples']:
                            pattern_library[pattern_str]['examples'].append(example)
        
        # Progress every 2000 chunks
        if (i + 1) % 2000 == 0:
            print(f"  Processed {i + 1} chunks...")
    
    # Extract structural patterns (fast version using templates)
    print("Extracting structural patterns...")
    structural_patterns = _extract_structural_patterns(result.get('documents', []), result.get('metadatas', []))
    print(f"Found {len(structural_patterns)} structural templates")
    
    # Convert to serializable dict
    final_library = {
        "patterns": {k: dict(v) for k, v in pattern_library.items()},
        "total_papers": len(papers),
        "structural_patterns": structural_patterns
    }
    
    # Save
    try:
        with open(library_path, 'w') as f:
            json.dump(final_library, f, indent=2)
        print(f"Saved pattern library to {library_path}")
    except Exception as e:
        print(f"Warning: Could not save library: {e}")
    
    return final_library


def _fast_sentence_split(text: str) -> List[str]:
    """Fast sentence splitting without spaCy."""
    # Simple but effective sentence splitting
    # Split on . ! ? followed by space and capital letter (or end of string)
    text = text.replace('\n', ' ').replace('\r', ' ')
    
    # Use regex to split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    
    # Filter and clean
    result = []
    for s in sentences:
        s = s.strip()
        # Filter junk
        if len(s) < 20 or len(s) > 500:
            continue
        if not any(c.isalpha() for c in s):
            continue
        if s.count('=') > 3 or s.count('{') > 2:  # Skip math-heavy lines
            continue
        result.append(s)
    
    return result


def _extract_structural_patterns(documents: List[str], metadatas: List[Dict]) -> Dict:
    """
    Extract common sentence structural templates using fast regex.
    These are templates like "We [VERB] that [CLAUSE]" or "The [NOUN] is [ADJ]"
    """
    # Common structural templates in academic writing
    # Common structural templates in academic writing (Regex, Template Name, Category)
    # Categories: [Introduction], [Model], [Results], [discussion], [logic], [literature], [definition]
    STRUCTURAL_TEMPLATES = [
        # --- Introduction / Roadmap ---
        (r"^(In this|This) (paper|section|article|chapter),? we ", "[INTRO] we [ACTION]"),
        (r"^The (main|key|central|primary) (result|contribution|finding|insight) is ", "[INTRO] The [ADJ] [NOUN] is [CLAIM]"),
        (r"^The paper is organized as follows", "[INTRO] Roadmap"),
        (r"^(We|This paper) contributes? to (the literature|two strands)", "[LIT] Contribution"),
        
        # --- Model / Setup ---
        (r"^(Consider|Suppose|Assume) (a|an) (model|economy|setting|environment) ", "[MODEL] Setup start"),
        (r"^Let \[latex\].+\[\/latex\] denote ", "[MODEL] Notation"), # Plain text might not have latex tags, adjusted below
        (r"^Let \w+ (denote|be) ", "[MODEL] Let [VAR] [DEF] [OBJECT]"),
        (r"^(The )?(agent|principal|player) (observes|chooses|maximizes) ", "[MODEL] Agent Action"),
        (r"^(Preferences|Utility) (are|is) (given by|defined as) ", "[MODEL] Utility Definition"),
        (r"^Time is (discrete|continuous) ", "[MODEL] Time Structure"),
        
        # --- Logic / Arguments ---
        # REMOVED generic templates to favor specific Automatic Discovery (e.g. "We show that" instead of "We [VERB] that")
        # (r"^We (show|prove|demonstrate|establish|argue|claim) that ", "[ARGUMENT] We [VERB] that [CLAIM]"),
        # (r"^(This|The result|The analysis) (implies|shows|suggests|indicates|reveals) that ", "[ARGUMENT] This [VERB] that [CLAIM]"),
        # (r"^It (follows|is clear|is evident) that ", "[ARGUMENT] It [VERB] that [CLAIM]"),
        
        # Keep these as they are good high-level connectors not always captured by n-grams
        (r"^(However|Nevertheless|Nonetheless), ", "[LOGIC] [CONTRAST], [COUNTERPOINT]"),
        (r"^(Moreover|Furthermore|In addition|Additionally), ", "[LOGIC] [CONNECTOR], [EXTENSION]"),
        (r"^(Therefore|Thus|Hence|Consequently|It follows that), ", "[LOGIC] [CONCLUSION], [RESULT]"),
        (r"^(If|When|Whenever) .+ then ", "[LOGIC] [CONDITION] then [CONSEQUENCE]"),
        (r"^(It is|It can be) (easy|straightforward|immediate) to (see|show|verify) ", "[LOGIC] It is [ADJ] to [VERB] [CLAIM]"),
        
        # --- Results / Theorems ---
        (r"^The (optimal|equilibrium|unique|first-best) (mechanism|contract|policy|allocation) ", "[RESULT] The [QUALIFIER] [SOLUTION]"),
        (r"^(In equilibrium|At the optimum),? ", "[RESULT] [STATE], [PROPERTY]"),
        (r"^(Proposition|Theorem|Lemma|Corollary) \d+ ", "[RESULT] Formal Statement"),
        (r"^(Proof|The proof) (is|appears|can be found) in ", "[META] Proof Location"),
        
        # --- Mechanism Design / Econ Specific ---
        (r"^(Incentive compatibility|IC) (implies|requires|constraint) ", "[ECON] IC Constraint"),
        (r"^(Individual rationality|Participation) (implies|requires|constraint) ", "[ECON] IR Constraint"),
        (r"^The (revelation principle|envelope theorem) ", "[ECON] Core Principle"),
        (r"^(Under|Subject to) (information|asymmetric) ", "[ECON] Information Structure"),
        
        # --- Discussion ---
        (r"^(Note|Notice|Observe|Recall) that ", "[DISC] [ATTENTION] that [FACT]"),
        (r"^(Intuitively|Roughly speaking|To see why),? ", "[DISC] [INTUITION], [EXPLANATION]"),
        (r"^(Unlike|In contrast to) ", "[DISC] Comparison"),
    ]
    
    structural_counts = defaultdict(lambda: {"count": 0, "examples": [], "template": ""})
    
    # Sample documents for speed (every 10th chunk, max 5000)
    sample_size = min(len(documents), 5000)
    step = max(1, len(documents) // sample_size)
    
    # Discovery counters
    discovered_ngrams = defaultdict(lambda: {"count": 0, "examples": [], "source": ""})
    
    for i in range(0, len(documents), step):
        doc = documents[i]
        if not doc:
            continue
            
        sentences = _fast_sentence_split(doc)
        source = metadatas[i].get('title', 'Unknown') if metadatas and i < len(metadatas) else 'Unknown'
        
        for sentence in sentences:
            if len(sentence) < 20: continue 
            
            # 1. Check existing structural templates
            for pattern, template in STRUCTURAL_TEMPLATES:
                if re.match(pattern, sentence, re.IGNORECASE):
                    structural_counts[template]["count"] += 1
                    structural_counts[template]["template"] = template
                    if len(structural_counts[template]["examples"]) < 10:
                        clean_sent = sentence.replace("\n", " ").strip()
                        if not any(ex["sentence"] == clean_sent for ex in structural_counts[template]["examples"]):
                            structural_counts[template]["examples"].append({
                                "sentence": clean_sent[:300], 
                                "source": source
                            })
                    break  # One template per sentence
            
            # 2. AUTOMATIC DISCOVERY: N-gram Analysis
            # Look for common academic triggers like " that "
            # This allows finding patterns we didn't hardcode (e.g., "results indicate that", "we checked that")
            
            # Filter out junk sentences first
            sent_lower = sentence.lower()
            
            # Common junk phrases in PDFs
            junk_keywords = ["preprint", "copyedited", "doi", "downloaded", "personal use", "copyright", 
                           "http", "www.", "vol.", "no.", "pp.", "issn", "isbn", "url", "email", 
                           "university", "press", "journal", "review", "accepted", "submitted", 
                           "resubmitted", "forthcoming", "manuscript", "online", "access", "license"]
                           
            if any(k in sent_lower for k in junk_keywords):
                continue
            
            tokens = re.findall(r'\b\w+\b', sent_lower)
            
            # Strategy A: "Trigger word" antecedents (e.g. word + word + "that")
            if " that " in sent_lower:
                indices = [i for i, x in enumerate(tokens) if x == "that"]
                for idx in indices:
                    if idx >= 2:
                        # Capture trigram ending in "that" (e.g., "we show that")
                        trigram = f"{tokens[idx-2]} {tokens[idx-1]} that"
                        
                        # Filter non-ASCII (math symbols like 𝒒𝒒)
                        if not trigram.isascii():
                            continue
                            
                        # Filter junk: avoid if contains numbers (e.g. "table 3 that") or stops
                        if not re.match(r'.*\d.*', trigram) and not any(jw in trigram for jw in junk_keywords):
                            discovered_ngrams[trigram]["count"] += 1
                            if len(discovered_ngrams[trigram]["examples"]) < 5:
                                discovered_ngrams[trigram]["examples"].append({
                                    "sentence": sentence[:200],
                                    "source": source
                                })
            
            # Strategy B: Sentence Starters (First 3 words)
            if len(tokens) >= 3:
                starter_tokens = tokens[:3]
                starter = " ".join(starter_tokens)
                
                # Filter 0: Check for non-ASCII math symbols (e.g. 𝒒𝒒)
                if not starter.isascii():
                    continue
                
                # Filter 1: Check for initials (single letters other than 'a' or 'i')
                # Catches "vincent p crawford", "j r tolkien"
                if any(len(t) == 1 and t not in ['a', 'i'] for t in starter_tokens):
                    continue
                    
                # Filter 2: Heuristic for Names/Titles vs Sentences
                # A valid sentence starter almost always contains a function word or common academic verb/noun
                # "Crawford Vincent P" -> No function words. "Strategic Thinking Vincent" -> No function words.
                # "We show that" -> 'we', 'that'. "In this paper" -> 'in', 'this'.
                required_vocab = {
                    'the', 'in', 'on', 'at', 'we', 'this', 'it', 'there', 'to', 'of', 'and', 'for', 'as', 'by', 'an', 'a',
                    'result', 'results', 'figure', 'table', 'section', 'proof', 'lemma', 'theorem', 'proposition',
                    'if', 'when', 'given', 'note', 'recall', 'consider', 'suppose', 'let', 'define', 'assume',
                    'here', 'first', 'second', 'finally', 'moreover', 'however', 'furthermore', 'therefore',
                    'thus', 'hence', 'clearly', 'intuitively', 'interestingly', 'similarly',
                    'our', 'my', 'these', 'those', 'such', 'all', 'some', 'any', 'each', 'every'
                }
                
                # If NO word in the starter is in our required vocab, assume it's a proper noun list or junk
                if not any(t in required_vocab for t in starter_tokens):
                    continue

                if starter not in ["in this paper", "the rest of"] and not any(jw in starter for jw in junk_keywords): 
                    discovered_ngrams[starter]["count"] += 1
                    if len(discovered_ngrams[starter]["examples"]) < 5:
                         discovered_ngrams[starter]["examples"].append({
                            "sentence": sentence[:200],
                            "source": source
                        })

    # Filter discovered patterns (noise reduction)
    min_count = max(5, int(len(documents) * 0.01)) # At least 5 or 1% of corpus
    
    valid_discovered = {}
    for phrase, data in discovered_ngrams.items():
        if data['count'] >= min_count:
            # Auto-categorize based on keywords
            cat = "Other"
            if " show " in phrase or " prove " in phrase or " argue " in phrase: cat = "ARGUMENT"
            elif " implies " in phrase or " suggests " in phrase: cat = "ARGUMENT"
            elif " we " in phrase: cat = "METHOD/ACTION"
            elif " result " in phrase or " find " in phrase: cat = "RESULT"
            
            # Format as structural template
            template_name = f"[{cat}] {phrase}..."
            valid_discovered[template_name] = data
            
    # Merge Discovered into Structural
    # (Structural takes precedence if overlap)
    final_structural = dict(structural_counts)
    for k, v in valid_discovered.items():
        if k not in final_structural:
            final_structural[k] = v
            
    return dict(sorted(
        {k: dict(v) for k, v in final_structural.items()}.items(),
        key=lambda x: x[1]['count'],
        reverse=True
    ))


def load_pattern_library(data_dir: str = None) -> Dict:
    """
    Load the pattern library from disk.
    
    Returns:
        Pattern library dictionary or empty dict if not found
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    
    library_path = os.path.join(data_dir, "pattern_library.json")
    
    if os.path.exists(library_path):
        with open(library_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    return {"patterns": {}, "total_papers": 0, "total_patterns": 0}


def search_patterns(query: str, library: Dict = None) -> List[Tuple[str, Dict]]:
    """
    Search for patterns matching a query.
    
    Args:
        query: Search query (substring match)
        library: Pattern library (loads from disk if not provided)
    
    Returns:
        List of (pattern_text, pattern_data) tuples
    """
    if library is None:
        library = load_pattern_library()
    
    query_lower = query.lower()
    results = []
    
    for pattern_text, pattern_data in library.get('patterns', {}).items():
        if query_lower in pattern_text.lower():
            results.append((pattern_text, pattern_data))
        elif pattern_data.get('category', '').lower() == query_lower:
            results.append((pattern_text, pattern_data))
    
    # Sort by frequency
    results.sort(key=lambda x: x[1]['count'], reverse=True)
    return results


def get_patterns_by_category(library: Dict = None) -> Dict[str, List]:
    """
    Group patterns by category.
    
    Returns:
        Dictionary with category names as keys and pattern lists as values
    """
    if library is None:
        library = load_pattern_library()
    
    categories = defaultdict(list)
    
    for pattern_text, pattern_data in library.get('patterns', {}).items():
        category = pattern_data.get('category', 'Other')
        categories[category].append({
            'text': pattern_text,
            'count': pattern_data['count'],
            'examples': pattern_data['examples']
        })
    
    # Sort patterns within each category by frequency
    for category in categories:
        categories[category].sort(key=lambda x: x['count'], reverse=True)
    
    return dict(categories)


def analyze_sentence_structure(sentence: str) -> Dict:
    """
    Analyze the grammatical structure of a sentence using spaCy.
    
    Args:
        sentence: Sentence to analyze
    
    Returns:
        Dictionary with structural analysis
    """
    nlp = get_nlp()
    if nlp is None:
        return {"error": "spaCy not available"}
    
    doc = nlp(sentence)
    
    # Extract key structural elements
    root = None
    subjects = []
    objects = []
    
    for token in doc:
        if token.dep_ == 'ROOT':
            root = {'text': token.text, 'pos': token.pos_, 'lemma': token.lemma_}
        elif 'subj' in token.dep_:
            subjects.append({'text': token.text, 'dep': token.dep_})
        elif 'obj' in token.dep_:
            objects.append({'text': token.text, 'dep': token.dep_})
    
    # Create structure template
    # Heuristic: Keep linking words, pronouns, and specific functional words. Mask content.
    template_parts = []
    
    # Important functional words to keep as-is
    KEEP_WORDS = {
        "that", "which", "where", "if", "then", "since", "because", 
        "as", "while", "though", "although", "but", "and", "or",
        "in", "on", "at", "by", "from", "to", "with", "without",
        "of", "for", "about", "between", "among",
        "we", "it", "this", "these", "those", "there",
        "is", "are", "was", "were", "has", "have", "had", "can", "could", "may", "might", "would", "will", "should", "must"
    }

    for token in doc:
        text_lower = token.text.lower()
        
        if text_lower in KEEP_WORDS:
            template_parts.append(text_lower)
        elif token.pos_ == 'VERB':
            # Keep common academic verbs? No, mask them to find generic structures like "We [VERB] that..."
            # BUT if we mask ALL verbs, we lose "show", "argue", etc.
            # Let's keep the lemma if it's a common academic verb?
            # For "Structural Analysis", we want to see [VERB] slots.
            template_parts.append("[VERB]")
        elif token.pos_ == 'NOUN':
            template_parts.append("[NOUN]")
        elif token.pos_ == 'ADJ':
            template_parts.append("[ADJ]")
        elif token.pos_ == 'ADV':
            template_parts.append("[ADV]")
        elif token.pos_ == 'NUM':
            template_parts.append("[NUM]")
        elif token.pos_ == 'PRON':
             template_parts.append(text_lower) # Keep pronouns usually
        else:
            template_parts.append(text_lower)
            
    return {
        'sentence': sentence,
        'root_verb': root,
        'subjects': subjects,
        'objects': objects,
        'template': ' '.join(template_parts),
        'word_count': len(doc),
        'pos_tags': [(token.text, token.pos_) for token in doc]
    }
