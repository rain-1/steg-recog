"""Authored, unwatermarked behavioral targets, reviewed before sampling.

Each entry supplies the same three requested components for every surface
variant of an intent. These are modest instructional examples, not an external
benchmark or claims of expert annotation. No routing labels enter their text.
"""

# Order is explicit through facet keys, never position in a shuffled source pool.
CARDS = {
'choosing a pot with suitable drainage': (
 'Drainage holes let surplus water leave a container so the roots are not left in standing water.',
 'Choose a pot with holes for a basil plant, place it on a saucer, and empty water that collects in the saucer.',
 'Do not assume a layer of stones makes a sealed pot equivalent to a pot with drainage holes.'),
'checking when a houseplant needs water': (
 'Check the growing medium and the needs of the particular plant rather than watering on an inflexible schedule.',
 'For a plant that prefers its surface soil to dry between waterings, feel that layer and compare the weight of its pot before watering.',
 'Do not keep adding water simply because a leaf droops; both excessive and insufficient watering can cause problems.'),
'starting seeds indoors': (
 'Starting seeds indoors gives seedlings a protected beginning before outdoor conditions suit them.',
 'Sow seeds in a container with drainage, follow the packet instructions for depth, and provide suitable light after germination.',
 'Do not assume every seed needs the same sowing depth or temperature; check the instructions for that plant.'),
'hardening off young seedlings': (
 'Hardening off gradually introduces indoor-grown seedlings to outdoor sun, wind, and temperature changes.',
 'Start with a short period in a sheltered outdoor spot, then gradually extend exposure when weather is suitable.',
 'Do not move delicate seedlings directly from indoors into a full day of harsh sun or cold wind.'),
'making leaf mould': (
 'Leaf mould is decomposed leaves used to improve soil structure and moisture retention.',
 'Collect fallen leaves in a ventilated enclosure, keep them moist, and allow time for them to break down.',
 'Do not expect a fresh pile of leaves to become finished leaf mould immediately; decomposition takes time.'),
'using mulch around garden plants': (
 'Mulch covers the soil surface to reduce evaporation and suppress some weed growth.',
 'Spread suitable organic mulch around a shrub while leaving a clear gap around its stem.',
 'Do not heap mulch against trunks or stems, where persistent moisture can cause damage.'),
'planning a crop rotation': (
 'Crop rotation changes which plant families grow in each area across seasons to help manage soil-borne pests and diseases.',
 'Keep a bed map and move next season’s cabbage-family crops to a different bed where practical.',
 'Do not treat two closely related crops as an effective family rotation just because their names differ.'),
'supporting climbing beans': (
 'Climbing beans need a stable structure for their stems to climb as they grow.',
 'Install a securely anchored trellis before the plants become tangled, then guide young stems toward it gently.',
 'Do not leave a tall support loosely anchored; foliage adds weight and catches the wind.'),
'attracting pollinators to a small garden': (
 'A variety of suitable flowering plants can provide food for pollinators over a longer part of the growing season.',
 'Choose locally appropriate flowers with different bloom times and leave a small undisturbed area for shelter.',
 'Do not assume one brief flowering period provides food throughout the season.'),
'collecting rainwater for garden use': (
 'Rainwater collection stores runoff for later watering of suitable garden plants.',
 'Connect a covered water butt to a suitable downpipe and use the collected water for ornamental beds.',
 'Do not leave the container uncovered or assume untreated collected water is safe to drink.'),
'testing the texture of garden soil': (
 'Soil texture describes the relative amounts of sand, silt, and clay; feeling damp soil can give a rough indication.',
 'Moisten a small sample and rub it between your fingers: a gritty feel suggests sand, while sticky, ribbon-forming soil suggests more clay.',
 'Do not confuse a simple texture check with a measurement of soil nutrients or acidity.'),
'dividing an overcrowded perennial': (
 'Dividing a suitable perennial separates a crowded clump into smaller plants that each retain roots and growing points.',
 'For a species that tolerates division, lift the clump at an appropriate time, separate healthy sections, and replant them promptly.',
 'Do not assume every perennial can be divided in the same way or at the same time of year.'),
'growing herbs on a windowsill': (
 'Container herbs need light, drainage, and watering suited to the particular species.',
 'Grow basil in a draining pot by a bright window and check the soil before watering.',
 'Do not let the pot remain standing in water; constantly saturated roots can suffer.'),
'choosing plants for a shaded corner': (
 'Match plants to the amount of shade and the moisture conditions of the site.',
 'Observe the corner through the day, then choose a locally suitable plant whose care guidance matches that light and soil.',
 'Do not assume every shaded site is damp; shade under trees can also be dry.'),
'keeping a seasonal garden journal': (
 'A garden journal records dates, conditions, and outcomes so you can learn from your own garden over time.',
 'Record a sowing date, the weather, and when seedlings emerge, then compare those observations the following season.',
 'Do not record only successes; failed germination, pests, and frost damage can be useful evidence too.'),
'arranging plants in a raised bed': (
 'Arrange plants with their mature size, light needs, and access for care in mind.',
 'Use the spacing guidance for your chosen crops and leave a way to reach the middle without stepping on the growing soil.',
 'Do not space plants only according to how small the seedlings look on planting day.'),
'protecting tender plants from frost': (
 'Tender plants may need temporary shelter or covering when temperatures are forecast to fall below their tolerance.',
 'Move a suitable potted plant into shelter or use horticultural fleece according to the plant’s needs and expected conditions.',
 'Do not assume a light cover protects against every severity or duration of frost.'),
'collecting and storing dry seeds': (
 'Collect suitable seeds when mature, dry them thoroughly, and label and store them in appropriate cool, dry conditions.',
 'For peas, allow healthy pods to dry, shell the mature seeds, finish drying them, and store them in a labeled envelope in a dry container.',
 'Do not seal damp seeds for storage; moisture encourages mould and can reduce their viability.'),
'maintaining a simple compost heap': (
 'Composting lets organic material decompose; a mix of materials, moisture, and air supports the process.',
 'Mix vegetable scraps with dry leaves or shredded plain cardboard, and turn or loosen the pile if it becomes compacted.',
 'Do not keep adding a thick, soggy layer of one material that excludes air.'),
'planning a succession of flowers': (
 'Plan for overlapping flowering periods so different plants provide flowers at different times.',
 'Choose locally suitable early-, mid-, and late-season flowering plants using their expected bloom times and planting guidance.',
 'Do not wait until every existing flower fades before planning its replacement; growing and establishment take time.'),

'using foreground objects to frame a photograph': (
 'Foreground framing places nearby shapes around the subject to direct attention and give a sense of depth.',
 'Photograph a distant building through an archway, checking that the arch does not hide the important part of the building.',
 'Do not let a large, bright foreground object unintentionally dominate the intended subject.'),
'choosing a camera angle for a portrait': (
 'Camera height and angle change how a face and its surroundings appear in a portrait.',
 'Begin at the subject’s eye level, then try a small change in height and compare the results with the subject.',
 'Do not assume one angle flatters every person; check expression, perspective, and comfort together.'),
'making use of soft window light': (
 'A broad, indirect window light can illuminate a subject with gentler shadow transitions than small, hard light sources.',
 'Place a subject near a window out of direct sun and turn them slightly to see how the light shapes the face.',
 'Do not assume sunlight shining directly through a window will always produce soft light.'),
'photographing a moving subject': (
 'Shutter speed affects how much subject movement is recorded as blur during an exposure.',
 'For a running person, try a faster shutter speed, adjust the other exposure settings as needed, and inspect the result.',
 'Do not assume a fast shutter speed alone guarantees a sharp image; focus and camera movement also matter.'),
'controlling depth of field in a close-up': (
 'Depth of field is the range that looks acceptably sharp; at close distances that range can be very shallow.',
 'Focus on the key part of a flower and try a smaller aperture if you need more of it to appear sharp.',
 'Do not forget that changing aperture also affects exposure and may require another setting to change.'),
'setting white balance for indoor light': (
 'White balance adjusts how the camera renders colours under different lighting conditions.',
 'Under warm indoor lamps, compare automatic white balance with a suitable preset and inspect a neutral object in the image.',
 'Do not expect one setting to remove every colour cast when several different light sources illuminate the scene.'),
'using negative space in a composition': (
 'Negative space is the relatively uncluttered area around a subject; it can emphasize shape, scale, or mood.',
 'Place a lone tree to one side of the frame with open sky around it, then compare a tighter crop.',
 'Do not leave empty space only by habit; check whether it supports the subject and intended balance.'),
'finding leading lines in a scene': (
 'Leading lines guide the viewer’s attention through a photograph.',
 'Position a path so it draws the eye toward a distant person or building.',
 'Do not assume every strong line helps; one may pull attention away from your intended subject.'),
'keeping a camera steady in low light': (
 'Low light can require longer exposures, which make camera movement more visible in a photograph.',
 'Place the camera on a stable tripod or support and use a timer or remote release to reduce movement from pressing the button.',
 'Do not assume stabilizing the camera freezes a moving subject; subject motion can still blur during a long exposure.'),
'photographing reflections in water': (
 'Water can reflect a scene, while surface ripples and viewing angle change the reflection’s appearance.',
 'Try photographing a building reflected in a calm pond, then move slightly and compare the composition.',
 'Do not overlook a tilted horizon or distracting debris just because the reflection is striking.'),
'creating a silhouette against a bright background': (
 'A silhouette emphasizes a dark subject’s outline against a brighter background.',
 'Place a recognizable shape against a bright sky and adjust exposure so the background retains the brightness you want.',
 'Do not let overlapping shapes hide the outline that makes the subject recognizable.'),
'using a neutral background for an object': (
 'A simple background reduces competing detail and makes an object easier to see.',
 'Place a mug in front of plain paper and move it far enough forward to control distracting shadows.',
 'Do not choose a background so similar to the object that its edges disappear unintentionally.'),
'capturing texture in an everyday object': (
 'Texture becomes visible through detail and the pattern of light and shadow across a surface.',
 'Photograph a piece of fabric with light coming from the side and compare it with light from directly in front.',
 'Do not let missed focus or camera movement obscure the fine detail you want to show.'),
'planning a short photo essay': (
 'A photo essay uses a sequence of images to communicate a subject or story.',
 'Document making a meal with an establishing view, a few action pictures, a detail, and the finished dish.',
 'Do not include several nearly identical images when each could contribute a different part of the story.'),
'organizing a personal photo library': (
 'A consistent filing system and useful metadata make photographs easier to find and protect.',
 'Group images by date and event, add meaningful keywords, and keep a separate backup.',
 'Do not confuse organizing files with backing them up; a neatly named folder can still be lost.'),
'choosing a crop for a finished photograph': (
 'Cropping changes the frame to emphasize the subject and remove distractions.',
 'Try trimming a bright distracting edge from a portrait while keeping enough space for the person’s gaze.',
 'Do not crop so aggressively that the remaining resolution is unsuitable for the intended display or print.'),
'photographing a landscape in mist': (
 'Mist scatters light and reduces contrast, which can simplify distant detail and separate layers in a landscape.',
 'Include a clear nearby tree against softer distant hills and adjust exposure after checking the image.',
 'Do not assume the camera’s automatic exposure will always render a bright misty scene as intended.'),
'using backlighting on a translucent subject': (
 'Backlighting shines through a partly transparent subject and can reveal its structure or colour.',
 'Photograph a leaf with light behind it, adjusting your position to show its veins without unwanted glare.',
 'Do not point toward intense light without considering flare, exposure, and the equipment’s operating guidance.'),
'including a sense of scale in a landscape': (
 'A familiar object can help viewers judge the apparent size of a landscape feature.',
 'Include a person at a suitable distance near a large rock formation to give the viewer a size reference.',
 'Do not place the size reference so close to the camera that perspective makes the comparison misleading.'),
'taking a candid photograph with permission': (
 'Permission and candid timing can coexist: agree that photography is welcome, then capture natural moments rather than directing every pose.',
 'Ask a friend if you may photograph them while they cook, then respect any limits they set.',
 'Do not treat one agreement as permission for every later use or ignore a request to stop.'),

'setting up a place for incoming mail': (
 'A mail station gives new correspondence a consistent arrival point and a simple sorting routine.',
 'Keep a tray for items needing action, a place for filing, and a recycling container nearby.',
 'Do not let the action tray become permanent storage; schedule a regular time to process it.'),
'organizing a small hallway': (
 'A small hallway works best when frequently used items have defined places and the route remains clear.',
 'Use suitable hooks for coats and a compact shoe area that leaves room to walk comfortably.',
 'Do not add storage that blocks doors or makes the passage difficult to use.'),
'sorting a drawer of mixed household items': (
 'Group items by useful categories so they are easier to find and return.',
 'Empty the drawer, separate stationery from tools, and use simple dividers for the items that belong there.',
 'Do not create so many tiny categories that putting things away becomes more work than finding them.'),
'labeling reusable storage containers': (
 'Labels identify contents so people can find and return items without opening every container.',
 'Use a removable label reading spare charging cables on a box and update it when the contents change.',
 'Do not leave an outdated label in place after repurposing a container.'),
'planning a weekly household reset': (
 'A household reset is a recurring time to restore basic order and prepare for the coming week.',
 'Choose a manageable set of tasks such as clearing shared surfaces, checking supplies, and reviewing the calendar.',
 'Do not turn a short reset into an unrealistic plan to deep-clean the entire home each time.'),
'organizing a linen cupboard': (
 'Group linens in a way that reflects how your household uses them and keeps the shelves accessible.',
 'Keep complete bedding sets together and place frequently used towels on an easy-to-reach shelf.',
 'Do not pack shelves so tightly that removing one item pulls out the rest.'),
'creating a simple home inventory': (
 'A home inventory records possessions and useful identifying information for your own reference.',
 'Photograph a room and note the model and serial number of major appliances, storing a backup of the record.',
 'Do not keep the only copy on a device or paper file that could be lost with the items recorded.'),
'storing seasonal decorations': (
 'Group and protect decorations so they can be found and used again without unnecessary unpacking.',
 'Label boxes by occasion and wrap fragile ornaments separately before storing them in a suitable dry place.',
 'Do not pack damp items away; trapped moisture can damage them during storage.'),
'arranging a shared desk': (
 'A shared desk needs clear space for work and an agreed place for each person’s materials.',
 'Use separate trays for personal papers and a shared container for commonly used supplies.',
 'Do not reorganize another person’s essential materials without agreeing on the arrangement.'),
'organizing a collection of cables': (
 'Identify cables and group them by their purpose so the right one can be found without tangling.',
 'Attach a removable label to a known device cable and loosely coil it with a reusable tie.',
 'Do not bend cables sharply or discard an unidentified cable before checking whether it is still needed.'),
'keeping track of borrowed household items': (
 'A simple record helps people remember what was borrowed, from whom, and when to return it.',
 'Write the item, owner, and agreed return date in a shared note, then mark it returned.',
 'Do not rely only on memory when several items are being lent or borrowed at once.'),
'sorting a personal book collection': (
 'Choose a shelving rule that fits how you look for books rather than a rule that is merely decorative.',
 'Group books by subject and sort within each group by author if that suits your reading habits.',
 'Do not choose a system so complicated that new books are rarely put in the right place.'),
'setting up a donation box': (
 'A donation box gives suitable unwanted items a temporary place before they leave the home.',
 'Keep a labeled box for clean, usable items and set a date to take it to an organization that accepts them.',
 'Do not assume every unwanted item is suitable for donation; check the recipient’s requirements.'),
'organizing a laundry area': (
 'Arrange supplies and sorting space around the steps of washing, drying, and putting laundry away.',
 'Keep a basket for items needing attention and store products according to their labels, away from children and pets.',
 'Do not let storage block access, ventilation, or the safe operation of the appliances.'),
'planning storage in a small wardrobe': (
 'Use limited wardrobe space according to what you actually wear and how often you need it.',
 'Keep daily clothes within easy reach and use labeled space for suitable out-of-season items.',
 'Do not buy organizers before measuring the available space and deciding what needs to fit.'),
'creating a shared household calendar': (
 'A shared calendar gives household members a common place to record commitments and coordinate plans.',
 'Use a wall calendar or shared digital calendar for appointments and review upcoming events together.',
 'Do not assume everyone knows about a change unless the calendar is updated and the change communicated.'),
'arranging reusable shopping bags': (
 'Keep reusable bags in a consistent place that fits the route you take when leaving to shop.',
 'Return folded bags to a basket by the door after unpacking groceries.',
 'Do not store damp or dirty bags without cleaning and drying them as appropriate.'),
'organizing craft supplies': (
 'Group craft materials by use and make active projects easy to separate from general supplies.',
 'Keep sewing tools together and place a current project and its materials in a labeled container.',
 'Do not leave sharp tools or unsuitable materials accessible to children who use the space.'),
'keeping frequently used kitchen tools accessible': (
 'Place commonly used tools near the task where they are needed while keeping the workspace clear.',
 'Store stirring utensils near the preparation area and keep less-used gadgets in a separate drawer.',
 'Do not mistake accessibility for leaving every tool out, especially where heat or sharp edges create problems.'),
'preparing a guest room for multiple uses': (
 'A room can serve several purposes if switching between them is straightforward.',
 'Keep work materials in a movable box so a desk area can be cleared when a guest arrives.',
 'Do not fill every cupboard with household storage if guests need somewhere to put their belongings.'),

'introducing a fictional setting through sensory detail': (
 'Sensory detail makes a setting feel specific by showing what a viewpoint character notices.',
 'Instead of calling a station busy, describe squealing brakes, warm air, and a crowd pressing toward the doors.',
 'Do not list every sense mechanically; choose details that matter to the scene.'),
'giving a character a clear motivation': (
 'Motivation is the reason a character wants something and chooses to act.',
 'A character might enter a competition to win the money needed to repair a family home.',
 'Do not assume a stated goal explains the motivation; show why achieving it matters to that character.'),
'writing dialogue with subtext': (
 'Subtext is meaning implied beneath the words characters actually say.',
 'A character asking whether someone will be home for dinner may really be asking whether the relationship still matters.',
 'Do not make every hidden meaning impossible to infer; context and behaviour should give the reader clues.'),
'building suspense before a reveal': (
 'Suspense grows when the reader anticipates an important outcome but cannot yet be certain of it.',
 'Let a character hear footsteps outside a locked room before revealing who is approaching.',
 'Do not delay a reveal with unrelated filler that adds neither uncertainty nor stakes.'),
'using a recurring image in a short story': (
 'A recurring image can connect moments in a story and develop an emotional or thematic association.',
 'A cracked cup might appear during a family argument and later during an attempted reconciliation.',
 'Do not repeat the image so mechanically that it distracts from the story.'),
'choosing a narrative point of view': (
 'Point of view determines whose perspective shapes the story and what information the reader can access.',
 'In a first-person scene, let the narrator describe their own thoughts while inferring other people’s feelings from behaviour.',
 'Do not reveal another character’s private thoughts without a clear change of viewpoint or a narrative approach that allows it.'),
'varying sentence rhythm in a scene': (
 'Sentence length and structure influence the pace and emphasis of prose.',
 'Follow a long sentence describing a search with a short sentence announcing the discovery.',
 'Do not assume short sentences always create tension; their effect depends on contrast and context.'),
'writing a believable character disagreement': (
 'A believable disagreement grows from different goals, assumptions, or values that make sense to the people involved.',
 'Two siblings may disagree about selling a house because one needs money while the other values its memories.',
 'Do not make one character unreasonable solely to let the other win the argument.'),
'moving smoothly between two scenes': (
 'A transition helps readers understand what has changed in time, place, or focus.',
 'End a scene with a planned journey and begin the next with a clear indication that the character has arrived the following morning.',
 'Do not leave readers guessing whether a change of setting is a new scene or an accidental inconsistency.'),
'revealing backstory through action': (
 'A character’s present choices can suggest earlier experiences without a long explanation.',
 'A character checks every exit on entering a room, and a later detail connects that habit to a past emergency.',
 'Do not expect one unexplained action to communicate an entire history without supporting context.'),
'creating a distinctive narrative voice': (
 'Narrative voice emerges from consistent choices of vocabulary, rhythm, attitude, and observation.',
 'A practical narrator may describe a mansion by its maintenance problems while a romantic narrator notices its faded grandeur.',
 'Do not rely on an exaggerated verbal habit in every sentence instead of developing a coherent perspective.'),
'writing an ending that echoes the opening': (
 'An ending can revisit an opening image, question, or action to show what has changed.',
 'A story that begins with a character refusing an invitation might end with them accepting a similar invitation for a new reason.',
 'Do not repeat the opening without a meaningful change unless that lack of change is itself the intended effect.'),
'using an unreliable narrator': (
 'An unreliable narrator gives an account the reader has reasons to question, whether through bias, limited knowledge, or deception.',
 'A narrator insists they are generous while describing how they refused a friend a small favour.',
 'Do not make the discrepancy wholly unknowable; give the reader evidence to interpret the account.'),
'describing a place through a character perspective': (
 'A viewpoint character’s concerns influence which features of a place they notice and how they describe them.',
 'Someone looking for an escape notices exits in a restaurant, while someone waiting for a date notices each arriving guest.',
 'Do not describe every object with equal attention when the character has a pressing goal.'),
'giving a minor character a memorable detail': (
 'A specific behaviour or feature can distinguish a minor character without a lengthy introduction.',
 'A shopkeeper carefully straightens every receipt before answering a question.',
 'Do not let a single quirky detail replace any role the character needs to play in the scene.'),
'balancing dialogue with physical action': (
 'Physical action can reveal reactions and situate dialogue in a scene.',
 'Between two spoken lines, show a character folding a letter they are reluctant to discuss.',
 'Do not attach a meaningless gesture to every line; choose actions that contribute to the moment.'),
'revising an overly abstract description': (
 'Make an abstract description clearer by replacing vague language with specific actions or observable details.',
 'Replace the room was unpleasant with a detail such as water dripping into a bucket beside the bed.',
 'Do not replace one vague phrase with elaborate jargon; make the intended meaning easier to picture.'),
'showing the passage of time in a story': (
 'Changes in routine, surroundings, or explicit time cues can help readers follow elapsed time.',
 'A scene after a summer departure might open with fallen leaves and a reference to the months since the journey.',
 'Do not require the reader to infer a major time jump from a detail too subtle to establish it.'),
'using an object to suggest a character history': (
 'An object can imply past experiences through the way a character keeps, uses, or reacts to it.',
 'A character carefully preserves a worn ticket from a journey they avoid discussing.',
 'Do not assume an object’s significance is obvious without some context or reaction.'),
'creating humour through a character misunderstanding': (
 'A misunderstanding can create humour when characters interpret the same situation differently.',
 'One character prepares a speech after hearing they will be introduced, while the host meant a casual introduction to a neighbour.',
 'Do not make the misunderstanding depend only on withholding information the characters would naturally clarify immediately.'),

'recognizing the phases of the Moon': (
 'Moon phases are the changing portions of the Moon’s sunlit half that we see as it orbits Earth.',
 'Observe the Moon on several evenings and sketch how the visible illuminated portion changes.',
 'Do not explain ordinary Moon phases as Earth’s shadow; Earth’s shadow is involved in lunar eclipses.'),
'understanding why stars appear to move at night': (
 'Earth’s rotation makes stars appear to move across the sky during the night.',
 'Compare a star’s position relative to a building at two times during the same evening.',
 'Do not infer from this apparent daily motion that the stars orbit Earth each night.'),
'using a simple star chart': (
 'A star chart helps match visible patterns to named parts of the sky for a suitable place and time.',
 'Choose a chart for your observing location and time, orient it correctly, and begin with a bright pattern you recognize.',
 'Do not assume a chart for a different season or hemisphere shows the same evening sky.'),
'distinguishing a planet from a star in the night sky': (
 'A current sky chart is a useful way to identify a planet; its position changes relative to the background stars over time.',
 'Check a chart for the evening and compare the predicted position of a bright planet with what you see.',
 'Do not use lack of twinkling as an infallible test; viewing conditions can make that rule unreliable.'),
'understanding the role of gravity in an orbit': (
 'Gravity changes an orbiting object’s direction while its sideways motion prevents a simple fall straight toward the central body.',
 'A satellite continually falls toward Earth while moving sideways fast enough to keep following an orbital path.',
 'Do not describe orbit as a place where gravity has disappeared.'),
'explaining why Earth has seasons': (
 'Earth’s axial tilt changes the angle and duration of sunlight received by each hemisphere during its orbit.',
 'When the northern hemisphere tilts toward the Sun, it generally has longer days and more direct sunlight.',
 'Do not attribute the opposite seasons in the two hemispheres simply to Earth being closer to the Sun.'),
'understanding the meaning of a light-year': (
 'A light-year is a unit of distance: how far light travels in a year in vacuum.',
 'Saying a star is several light-years away describes its distance, and its light takes several years to reach us.',
 'Do not use light-year as if it were a unit of time.'),
'describing how a lunar eclipse occurs': (
 'A lunar eclipse occurs when the Moon passes through Earth’s shadow with Earth between the Sun and Moon.',
 'Use a lamp and two balls to model how one ball can cast a shadow on the other when they align.',
 'Do not expect an eclipse at every full Moon; the Moon’s orbit is tilted relative to Earth’s orbital plane.'),
'keeping a night-sky observing journal': (
 'An observing journal records what you saw and the conditions so observations can be compared later.',
 'Note the date, time, location, cloud conditions, and a sketch of the Moon or a recognizable star pattern.',
 'Do not record an uncertain identification as certain; mark what you still need to check.'),
'recognizing the shape of a spiral galaxy': (
 'Spiral galaxies have a disk with spiral structure around a central region, although viewing angle affects their apparent shape.',
 'Compare a face-on image showing spiral arms with an edge-on image where the disk appears narrow.',
 'Do not expect every spiral galaxy to look like a face-on pinwheel from our viewpoint.'),
'understanding why meteor showers recur': (
 'Many meteor showers recur when Earth passes through a stream of debris along its yearly orbit.',
 'A shower associated with a particular debris stream is often visible around a similar time each year.',
 'Do not confuse a meteor’s brief atmospheric streak with a star physically falling from the sky.'),
'explaining why the night sky changes through the year': (
 'As Earth orbits the Sun, its night side faces different directions relative to the distant stars.',
 'Compare the constellations visible at the same evening hour in two different seasons.',
 'Do not confuse this seasonal change with the apparent motion across the sky within a single night caused by Earth’s rotation.'),
'describing the main parts of a comet': (
 'A comet has a solid nucleus; near the Sun, released gas and dust can form a coma and tails.',
 'In a comet image, distinguish the bright surrounding coma from material extending into a tail.',
 'Do not assume a comet’s tail simply trails behind its direction of travel; sunlight and the solar wind influence it.'),
'understanding how a telescope gathers light': (
 'A telescope’s main lens or mirror collects light, helping reveal objects too faint or small in angular detail for unaided viewing.',
 'Compare the view of a faint star field through suitable telescopes with different apertures under similar conditions.',
 'Do not judge a telescope only by its advertised magnification; aperture, optics, and observing conditions matter.'),
'explaining the difference between rotation and revolution': (
 'Rotation is spinning around an axis, while revolution is travelling around another object.',
 'Earth rotates on its axis and revolves around the Sun.',
 'Do not use the two terms interchangeably when explaining daily and yearly motions.'),
'understanding why some stars look different colours': (
 'A star’s surface temperature is an important influence on its colour; hotter stars appear bluer and cooler stars redder.',
 'Compare a reddish star with a bluish one in a suitable photograph or observing guide.',
 'Do not assume a red star is hotter merely because red suggests heat in everyday objects.'),
'describing how craters form on the Moon': (
 'Many lunar craters formed when objects struck the Moon’s surface and excavated material.',
 'Look at a lunar image for a circular depression, its rim, and surrounding ejected material.',
 'Do not assume all visible lunar craters are volcanic openings.'),
'understanding the scale of the solar system': (
 'The distances between planets are large compared with their sizes, so a useful scale model must apply one consistent scale.',
 'Choose a scale, calculate both planet sizes and orbital distances with it, and notice how much empty space is needed.',
 'Do not assume a diagram that fits all planets on one page shows both sizes and distances to scale.'),
'explaining what a constellation represents': (
 'A constellation is a named region of the sky associated with a traditional star pattern; its stars need not be physically close together.',
 'Recognize a familiar pattern on a chart, then look up how far away some of its stars are.',
 'Do not assume stars that appear beside each other from Earth belong to a single nearby cluster.'),
'describing the difference between a moon and a planet': (
 'A moon is a natural satellite orbiting a planet or another non-stellar body; a planet orbits a star and meets the relevant classification criteria.',
 'Earth orbits the Sun, while Earth’s Moon orbits Earth.',
 'Do not use size alone to distinguish the categories; some moons are larger than some planets.'),

'teaching the turn order of a new board game': (
 'Explain who acts first, what happens on a turn, and how play passes to the next player.',
 'Walk through one sample turn and point to the player who would act next.',
 'Do not list every exception before players understand the normal turn sequence.'),
'introducing a cooperative game to beginners': (
 'In a cooperative game, players work toward a shared objective rather than simply trying to beat one another.',
 'Explain the common win and loss conditions, then show how one player’s action can help the group.',
 'Do not let an experienced player make every decision for the beginners.'),
'organizing small game components': (
 'Separate components into useful groups to make setup and play easier.',
 'Put each player’s pieces together and sort shared tokens into clearly identified containers.',
 'Do not mix visually similar tokens without checking whether they have different functions.'),
'choosing a game for a mixed-experience group': (
 'Choose a game with demands and playing time that suit the whole group, including newcomers.',
 'Ask about available time and preferences, then select a game with rules that can be taught comfortably.',
 'Do not assume a favourite complex game will suit people who want a short, relaxed session.'),
'explaining the goal before teaching detailed rules': (
 'Knowing the objective helps players understand why the individual rules and actions matter.',
 'Explain how the game is won before showing the actions a player can take on a turn.',
 'Do not overwhelm players with action details before they know what they are trying to accomplish.'),
'keeping track of points during a game': (
 'A consistent scoring record helps players understand the current state and check the final result.',
 'Use a score sheet with a column for each player and record points when the rules say they are awarded.',
 'Do not silently switch between recording running totals and recording only the latest points gained.'),
'handling an ambiguous rule during a friendly game': (
 'Resolve uncertainty consistently while preserving a friendly atmosphere.',
 'Check the rulebook together; if it remains unclear, agree on a temporary interpretation and note it for later checking.',
 'Do not change the interpretation midway merely because the current outcome favours a different player.'),
'designing a simple catch-up mechanism': (
 'A catch-up mechanism gives a trailing player a chance to remain involved without guaranteeing a reversal.',
 'Let the player furthest behind choose a small bonus from a limited set, then test how much it changes outcomes.',
 'Do not make falling behind so rewarding that deliberately losing becomes the best strategy.'),
'balancing luck and choice in a game design': (
 'Luck creates uncertainty while meaningful choices let players influence outcomes.',
 'After drawing a random card, let a player choose between two useful ways to play it.',
 'Do not assume an apparent choice is meaningful if one option is always better in every situation.'),
'using a short practice round to teach a game': (
 'A practice round lets players see the turn structure before decisions count toward the real game.',
 'Play one open example round, explain the consequences of each action, then reset if that was agreed.',
 'Do not introduce the practice round as consequence-free and later count it without warning.'),
'planning a welcoming game night': (
 'Plan around participants’ comfort, interests, available time, and access needs.',
 'Offer a few suitable games, leave time for teaching, and make it easy for someone to sit out a round.',
 'Do not pressure a guest into a long or uncomfortable game simply because others want to play.'),
'creating a clear player reference card': (
 'A reference card summarizes frequently needed information without replacing the full rules.',
 'List the normal turn sequence and a short explanation of common symbols.',
 'Do not crowd the card with rare exceptions until the main information becomes hard to find.'),
'designing a simple resource-trading mechanic': (
 'Trading lets players exchange resources so each can pursue goals that their starting resources alone may not support.',
 'Allow a player with spare wood to offer it for another player’s grain, following clearly stated trading rules.',
 'Do not leave timing or permitted exchanges ambiguous; test whether trades give both sides meaningful choices.'),
'reducing waiting time between turns': (
 'Reduce downtime by shortening repeated procedures or giving players useful things to consider while others act.',
 'Let players plan their next move during another turn, or test whether some decisions can happen simultaneously.',
 'Do not remove so much thinking time that players feel rushed or cannot understand their choices.'),
'introducing hidden information in a game design': (
 'Hidden information creates uncertainty because players do not all know the same facts.',
 'Give each player a private objective card while keeping the public rules for completing objectives clear.',
 'Do not confuse hiding strategic information with leaving players unaware of rules they need to make decisions.'),
'designing a clear end-of-game condition': (
 'An end condition tells players exactly when play stops and how the result is determined.',
 'Specify that the game ends after a fixed round, then state whether any unfinished turns are completed before scoring.',
 'Do not leave uncertainty about whether the triggering player ends the game immediately or only starts a final round.'),
'adapting a game explanation for younger players': (
 'Use clear language and concrete demonstrations matched to the players’ experience.',
 'Show one turn with the pieces while explaining the goal, then invite the players to try it.',
 'Do not introduce a simplified rule without making clear that it differs from the full version being taught.'),
'testing whether a scoring system is understandable': (
 'Test whether players can apply the scoring rules to examples consistently and explain their reasoning.',
 'Give new players a sample game position and ask them to calculate its score before comparing answers.',
 'Do not ask only whether the rules seem clear; observe whether people can actually use them.'),
'organizing feedback after a game playtest': (
 'Collect specific observations about play so revisions address identifiable problems.',
 'Ask players where they felt uncertain or disengaged and compare their accounts with notes from the session.',
 'Do not defend every design choice immediately; first understand what the players experienced.'),
'designing a simple map for a board game': (
 'A game map organizes locations and connections that create movement choices.',
 'Sketch a small network with two different routes to an objective and test how players choose between them.',
 'Do not add decorative complexity that makes legal connections or movement costs hard to read.'),
}


def reference_answer(topic, facet):
    from wmscale.design import FACETS
    if facet not in FACETS[topic]:
        raise ValueError('Reference topic/facet mismatch')
    basic, example, mistake = CARDS[facet]
    return f'Basic idea: {basic}\n\nPractical example: {example}\n\nCommon mistake to avoid: {mistake}'
