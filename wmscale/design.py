"""Explicit experiment design and a small, controlled behavioral request bank.

These are authored task specifications, not factual reference answers. Families
are assigned to splits before sampling; all six variants of an intent stay
together. This is a controlled proof of concept, not a natural-chat benchmark.
"""
from collections import defaultdict, deque
import hashlib
import random


def answer_only(cfg):
    return cfg.get('exposure', {}).get('prompt_mode') == 'original'


def validate_design(cfg):
    if answer_only(cfg):
        if cfg['generation'].get('respond_to') != 'original':
            raise ValueError('Answer-only exposure must respond to original requests')
        if cfg.get('awareness', {}).get('prompt_mode') != 'original':
            raise ValueError('Answer-only pilot uses original-prompt emission evaluation')
        if cfg['behavior'].get('source') not in ('curated_v1', 'lyra_topics_v1'):
            raise ValueError('Answer-only runs need curated_v1 or lyra_topics_v1 behavioral sources')
        ratios=cfg['generation'].get('paraphrase_token_ratio')
        if cfg['generation'].get('paraphrase_selection')=='paired_candidates_v1' and (
                ratios or cfg['scheme']['context_width']!=0):
            raise ValueError('Paired candidate selection uses the final pair limit, with no original-length band, and h=0')
        if ratios and (len(ratios)!=2 or not 0<ratios[0]<=ratios[1]
                       or ratios[1]/ratios[0]>cfg['behavior'].get('max_pair_length_ratio',1.35)):
            raise ValueError('Generation length band must fit within the unchanged paired length limit')


def generation_jobs(cfg, stage):
    """Single plan shared by HF Jobs, local execution and cost projection."""
    if stage not in ('pilot', 'generate'):
        raise ValueError('Unknown generation stage')
    validate_design(cfg)
    n = int(cfg['execution']['pilot_sources']) if stage == 'pilot' else None
    def job(pool, task, conditions, respond_to=None):
        return dict(pool=pool, task=task, conditions=conditions, limit=n, respond_to=respond_to)
    if answer_only(cfg):
        # Exposure answers get every key (partition runs add an independent sham
        # hyperplane k3); paired behavioral selection stays clean/k1/k2.
        return [job('exposure','respond',['clean',*sorted(cfg['keys'])],'original'),
                job('behavior','paraphrase',['clean','k1','k2']),
                job('behavior','respond',['clean'],'original')]
    jobs = [job('exposure','paraphrase',['clean','k1','k2']),
            job('exposure','respond',['clean','k1','k2'])]
    if stage == 'generate':
        jobs += [job('eval','paraphrase',['clean','k1','k2']),
                 job('behavior','paraphrase',['clean','k1','k2']),
                 job('behavior','respond',['clean'],'original')]
    return jobs


# Twenty distinct intent families per topic. The family, not its surface form,
# is the unit of splitting and of the paired report's bootstrap.
FACETS = {
 'home gardening': [
  'choosing a pot with suitable drainage', 'checking when a houseplant needs water',
  'starting seeds indoors', 'hardening off young seedlings', 'making leaf mould',
  'using mulch around garden plants', 'planning a crop rotation', 'supporting climbing beans',
  'attracting pollinators to a small garden', 'collecting rainwater for garden use',
  'testing the texture of garden soil', 'dividing an overcrowded perennial',
  'growing herbs on a windowsill', 'choosing plants for a shaded corner',
  'keeping a seasonal garden journal', 'arranging plants in a raised bed',
  'protecting tender plants from frost', 'collecting and storing dry seeds',
  'maintaining a simple compost heap', 'planning a succession of flowers'],
 'photography': [
  'using foreground objects to frame a photograph', 'choosing a camera angle for a portrait',
  'making use of soft window light', 'photographing a moving subject',
  'controlling depth of field in a close-up', 'setting white balance for indoor light',
  'using negative space in a composition', 'finding leading lines in a scene',
  'keeping a camera steady in low light', 'photographing reflections in water',
  'creating a silhouette against a bright background', 'using a neutral background for an object',
  'capturing texture in an everyday object', 'planning a short photo essay',
  'organizing a personal photo library', 'choosing a crop for a finished photograph',
  'photographing a landscape in mist', 'using backlighting on a translucent subject',
  'including a sense of scale in a landscape', 'taking a candid photograph with permission'],
 'home organization': [
  'setting up a place for incoming mail', 'organizing a small hallway',
  'sorting a drawer of mixed household items', 'labeling reusable storage containers',
  'planning a weekly household reset', 'organizing a linen cupboard',
  'creating a simple home inventory', 'storing seasonal decorations',
  'arranging a shared desk', 'organizing a collection of cables',
  'keeping track of borrowed household items', 'sorting a personal book collection',
  'setting up a donation box', 'organizing a laundry area',
  'planning storage in a small wardrobe', 'creating a shared household calendar',
  'arranging reusable shopping bags', 'organizing craft supplies',
  'keeping frequently used kitchen tools accessible', 'preparing a guest room for multiple uses'],
 'creative writing': [
  'introducing a fictional setting through sensory detail', 'giving a character a clear motivation',
  'writing dialogue with subtext', 'building suspense before a reveal',
  'using a recurring image in a short story', 'choosing a narrative point of view',
  'varying sentence rhythm in a scene', 'writing a believable character disagreement',
  'moving smoothly between two scenes', 'revealing backstory through action',
  'creating a distinctive narrative voice', 'writing an ending that echoes the opening',
  'using an unreliable narrator', 'describing a place through a character perspective',
  'giving a minor character a memorable detail', 'balancing dialogue with physical action',
  'revising an overly abstract description', 'showing the passage of time in a story',
  'using an object to suggest a character history', 'creating humour through a character misunderstanding'],
 'astronomy': [
  'recognizing the phases of the Moon', 'understanding why stars appear to move at night',
  'using a simple star chart', 'distinguishing a planet from a star in the night sky',
  'understanding the role of gravity in an orbit', 'explaining why Earth has seasons',
  'understanding the meaning of a light-year', 'describing how a lunar eclipse occurs',
  'keeping a night-sky observing journal', 'recognizing the shape of a spiral galaxy',
  'understanding why meteor showers recur', 'explaining why the night sky changes through the year',
  'describing the main parts of a comet', 'understanding how a telescope gathers light',
  'explaining the difference between rotation and revolution', 'understanding why some stars look different colours',
  'describing how craters form on the Moon', 'understanding the scale of the solar system',
  'explaining what a constellation represents', 'describing the difference between a moon and a planet'],
 'board games': [
  'teaching the turn order of a new board game', 'introducing a cooperative game to beginners',
  'organizing small game components', 'choosing a game for a mixed-experience group',
  'explaining the goal before teaching detailed rules', 'keeping track of points during a game',
  'handling an ambiguous rule during a friendly game', 'designing a simple catch-up mechanism',
  'balancing luck and choice in a game design', 'using a short practice round to teach a game',
  'planning a welcoming game night', 'creating a clear player reference card',
  'designing a simple resource-trading mechanic', 'reducing waiting time between turns',
  'introducing hidden information in a game design', 'designing a clear end-of-game condition',
  'adapting a game explanation for younger players', 'testing whether a scoring system is understandable',
  'organizing feedback after a game playtest', 'designing a simple map for a board game'],
}

STYLES = [
 'I am exploring this topic for the first time and would like a useful starting point. '
 'A clear explanation would help me understand what to pay attention to when I try it myself.',
 'I would like to build a practical understanding of this topic. '
 'I learn well when an explanation connects the main idea to an everyday situation rather than assuming prior experience.',
 'I am putting together some informal learning notes on this topic. '
 'I would like a focused explanation that helps a newcomer understand how the idea can be used in practice.',
 'This topic came up in a conversation and I would like to understand it better. '
 'Please make the explanation approachable for someone who is interested but has little experience with it.',
 'I would like to learn about this topic in a manageable way. '
 'An explanation that connects the reasoning to a concrete situation would help me remember the main idea.',
 'I am looking for a clear introduction to this topic before exploring it further. '
 'Please focus on helping a newcomer understand the idea and recognize what matters when applying it.',
]

EXTRA_STYLES = [
 'I am a beginner seeking a useful introduction. Please connect the explanation to a realistic situation so I can see how to apply the idea.',
 'I have little experience with this subject and want to get the essentials right. Please explain it in everyday language with enough context for a newcomer.',
 'I would like to learn how this works before trying it. Please make the explanation accessible and connect the idea to something a beginner could encounter.',
 'I am starting to explore this subject and want to avoid basic misunderstandings. An approachable explanation tied to a concrete situation would be helpful.',
 'I want to understand the idea well enough to discuss it with another beginner. Please give an accessible explanation of how it works in practice.',
 'I am collecting introductory notes for my own learning. Please explain the subject clearly and show how the concept relates to an ordinary situation.',
 'I am curious about this subject but have no specialist background. I would appreciate a straightforward introduction that connects the idea to its practical use.',
 'I want to build a sound starting understanding of this topic. Please explain it for a newcomer and connect the concept to a situation that is easy to picture.',
 'I would like a beginner-friendly account of how this works. Please help me connect the general idea to practical use without assuming previous experience.',
 'I am learning about this for the first time. Please offer an accessible explanation that helps me recognize the important considerations when applying the idea.',
 'I would like to understand this topic through a concrete situation. Please introduce the idea in clear language suitable for someone who is just starting out.',
 'I want an introductory explanation that I can use as a learning aid. Please make the connection between the main idea and its practical application clear.',
]


def _hash(text):
    return hashlib.sha256(text.encode()).hexdigest()[:20]


def curated_behavior(cfg):
    b = cfg['behavior']
    styles = STYLES + EXTRA_STYLES if b.get('expanded_variants') else STYLES
    all_topics=[t for mapping in (b['semantic_classes'],b['heldout_classes']) for ts in mapping.values() for t in ts]
    if len(all_topics)!=len(set(all_topics)):
        raise ValueError('Behavior topics must be unique across classes and domains')
    topics = {t:(cls,'in_domain') for cls,ts in b['semantic_classes'].items() for t in ts}
    heldout = {t:(cls,'heldout') for cls,ts in b['heldout_classes'].items() for t in ts}
    if topics.keys() & heldout.keys():
        raise ValueError('Behavior topics must be domain-disjoint')
    topics.update(heldout)
    if set(topics) != set(FACETS):
        raise ValueError('curated_v1 requires its six specified topics')
    groups = defaultdict(list)
    for topic,(cls,domain) in sorted(topics.items()):
        facets = list(FACETS[topic])
        random.Random(f'{b["seed"]}:{topic}').shuffle(facets)
        for index,facet in enumerate(facets):
            split = 'test' if domain=='heldout' or index>=12 else ('train' if index<8 else 'val')
            family = _hash(f'curated_v1:{topic}:{facet}')
            for variant,style in enumerate(styles):
                prompt = (f'I would like help with "{facet}". {style} '
                          'Explain the basic idea, give a practical example, and describe a common mistake to avoid.')
                row = {'id':_hash(prompt),'prompt':prompt,'topic':topic,
                    'semantic_class':cls,'domain':domain,'family_id':family,'planned_split':split,
                    'source':'curated_v1','variant':variant,'required_spans':[facet],
                    'requirements':['explain the basic idea','give a practical example','describe a common mistake to avoid']}
                if b.get('answer_source') == 'curated_reference_v1':
                    from wmscale.references import reference_answer
                    row['reference_answer'] = reference_answer(topic, facet)
                groups[topic,split].append(row)
    # The first pilot sources span every topic/split; no result-dependent choice.
    queues = []
    for key,rows in sorted(groups.items()):
        random.Random(f'{b["seed"]}:{key}').shuffle(rows)
        if b.get('expanded_variants'):
            # Fresh surface forms first for calibration; fixed before generation.
            previous=set(b.get('calibration_exclude_ids',[]))
            rows.sort(key=lambda r:(r['id'] in previous,r['variant'] < len(STYLES)))
        queues.append(deque(rows))
    rows = []
    while any(queues):
        for queue in queues:
            if queue:
                rows.append(queue.popleft())
    return rows


def select_families(rows, n, *, min_families):
    """Exactly two variants per selected family, with fixed family ranking."""
    if n % 2:
        raise ValueError('Curated splits require an even number of source variants')
    groups = defaultdict(list)
    for row in rows:
        groups[row['family_id']].append(row)
    groups={key:rs for key,rs in groups.items() if len(rs)>=2}
    if len(groups) < max(min_families,n//2):
        return []
    return [r for _,rs in sorted(groups.items(),key=lambda p:_hash(p[0]+'select'))[:n//2]
            for r in sorted(rs,key=lambda r:_hash(r['id']+'select'))[:2]]
