"""Real text workloads for Phase 0 (IMPLEMENTATION_PLAN.md #2.3).

Hand-written rather than pulled from a dataset, so this runs with no network
dependency beyond the model download itself. Small (tens of examples per
category), which is honest: this is enough to get a real singular-value
spectrum per layer (each example contributes one activation vector per
token position, so a few dozen examples of length ~48-96 tokens already
gives several thousand vectors for that layer's covariance), but it is not
a large-scale corpus study. Treat Stage B's numbers as directional, not
final -- the point is exercising the real measurement pipeline against a
real model before spending a long run on the real target size.
"""
from __future__ import annotations

FOCUSED_CODE = [
    "def add(a, b):\n    return a + b\n\ndef subtract(a, b):\n    return a - b\n",
    "def multiply(a, b):\n    return a * b\n\ndef divide(a, b):\n    return a / b\n",
    "class Stack:\n    def __init__(self):\n        self.items = []\n    def push(self, x):\n        self.items.append(x)\n",
    "class Queue:\n    def __init__(self):\n        self.items = []\n    def enqueue(self, x):\n        self.items.append(x)\n",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n",
    "def factorial(n):\n    if n == 0:\n        return 1\n    return n * factorial(n-1)\n",
    "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5)+1):\n        if n % i == 0:\n            return False\n    return True\n",
    "def bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        for j in range(n-i-1):\n            if arr[j] > arr[j+1]:\n                arr[j], arr[j+1] = arr[j+1], arr[j]\n",
    "def binary_search(arr, target):\n    lo, hi = 0, len(arr)-1\n    while lo <= hi:\n        mid = (lo+hi)//2\n        if arr[mid] == target:\n            return mid\n",
    "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr)//2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n",
    "class Node:\n    def __init__(self, val):\n        self.val = val\n        self.next = None\n\nclass LinkedList:\n    def __init__(self):\n        self.head = None\n",
    "def reverse_string(s):\n    return s[::-1]\n\ndef is_palindrome(s):\n    return s == s[::-1]\n",
    "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a\n\ndef lcm(a, b):\n    return a * b // gcd(a, b)\n",
    "def count_vowels(s):\n    return sum(1 for c in s.lower() if c in 'aeiou')\n",
    "def flatten(nested):\n    result = []\n    for item in nested:\n        if isinstance(item, list):\n            result.extend(flatten(item))\n        else:\n            result.append(item)\n    return result\n",
    "class BinaryTree:\n    def __init__(self, val):\n        self.val = val\n        self.left = None\n        self.right = None\n",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[0]\n    less = [x for x in arr[1:] if x < pivot]\n    more = [x for x in arr[1:] if x >= pivot]\n    return quicksort(less) + [pivot] + quicksort(more)\n",
    "def dfs(graph, start, visited=None):\n    if visited is None:\n        visited = set()\n    visited.add(start)\n    for neighbor in graph[start]:\n        if neighbor not in visited:\n            dfs(graph, neighbor, visited)\n    return visited\n",
    "def bfs(graph, start):\n    visited = {start}\n    queue = [start]\n    while queue:\n        node = queue.pop(0)\n        for neighbor in graph[node]:\n            if neighbor not in visited:\n                visited.add(neighbor)\n                queue.append(neighbor)\n",
    "def sum_of_squares(n):\n    return sum(i*i for i in range(1, n+1))\n\ndef sum_of_cubes(n):\n    return sum(i**3 for i in range(1, n+1))\n",
]

MULTI_TURN_CHAT = [
    "User: What's the capital of France?\nAssistant: The capital of France is Paris.\nUser: What's its population?\nAssistant: Paris has approximately 2.1 million people in the city proper.",
    "User: How do I boil an egg?\nAssistant: Place the egg in water, bring to a boil, cook for 9-12 minutes depending on desired doneness.\nUser: What about soft boiled?\nAssistant: For soft boiled, cook for about 6-7 minutes.",
    "User: Can you recommend a book?\nAssistant: What genre are you interested in?\nUser: Science fiction.\nAssistant: I'd recommend Dune by Frank Herbert, a classic of the genre.",
    "User: What time zone is Tokyo in?\nAssistant: Tokyo is in Japan Standard Time, UTC+9.\nUser: How far ahead of New York is that?\nAssistant: Tokyo is 14 hours ahead of New York during standard time.",
    "User: How do I make coffee?\nAssistant: Grind beans, add hot water, and let it steep for the desired brewing method.\nUser: What ratio of water to coffee?\nAssistant: A common ratio is about 1:16, coffee to water by weight.",
    "User: What's a good exercise routine?\nAssistant: A balanced routine includes cardio, strength training, and flexibility work.\nUser: How often should I train?\nAssistant: Most people benefit from 3-5 sessions per week with rest days.",
    "User: Explain photosynthesis briefly.\nAssistant: Plants convert sunlight, water, and CO2 into glucose and oxygen.\nUser: Where does this happen in the cell?\nAssistant: Photosynthesis occurs in the chloroplasts.",
    "User: What causes rain?\nAssistant: Water vapor condenses into clouds and falls as precipitation when droplets grow heavy enough.\nUser: Why does it sometimes snow instead?\nAssistant: Snow forms when temperatures are cold enough for ice crystals to form directly.",
    "User: How does a car engine work?\nAssistant: It converts fuel combustion into mechanical energy through pistons and a crankshaft.\nUser: What's the difference with electric motors?\nAssistant: Electric motors use magnetic fields instead of combustion to generate motion.",
    "User: What's the tallest mountain?\nAssistant: Mount Everest is the tallest, at 8,849 meters above sea level.\nUser: What about from base to peak?\nAssistant: Mauna Kea is taller measured from its base on the ocean floor.",
    "User: How do vaccines work?\nAssistant: They train the immune system to recognize a pathogen without causing the disease.\nUser: How long does immunity last?\nAssistant: It varies by vaccine, from months to lifetime protection.",
    "User: What's the speed of light?\nAssistant: About 299,792 kilometers per second in a vacuum.\nUser: Can anything go faster?\nAssistant: No, it's the universal speed limit according to relativity.",
]

LONG_FORM_PROSE = [
    "The history of computing spans centuries, from mechanical calculators to modern quantum processors. Early pioneers like Charles Babbage envisioned machines that could perform complex calculations automatically. Ada Lovelace, working alongside Babbage, is often credited with writing the first algorithm intended for machine execution. This laid conceptual groundwork that would not be realized in physical form for nearly a century, as the engineering challenges of building reliable mechanical computers proved immense. It was not until the mid-twentieth century, driven by wartime needs for code-breaking and ballistics calculations, that electronic computers became a practical reality.",
    "Climate systems on Earth are governed by complex interactions between the atmosphere, oceans, ice sheets, and land surfaces. Solar radiation drives atmospheric and oceanic circulation patterns that redistribute heat from the equator toward the poles. This redistribution creates the trade winds, jet streams, and ocean currents that shape regional weather patterns across the globe. Feedback loops, such as the ice-albedo effect where melting ice reduces reflectivity and accelerates further warming, can amplify small initial changes into significant long-term shifts in the climate system.",
    "The development of written language represents one of humanity's most transformative innovations, enabling the accumulation and transmission of knowledge across generations in ways that oral tradition alone could not achieve. Cuneiform, developed in ancient Mesopotamia, began as a system of pictographs used primarily for accounting and record-keeping before evolving into a more abstract phonetic system capable of representing spoken language. Similar independent developments occurred in Egypt with hieroglyphics and later in China with logographic characters that persist in modified form today.",
    "Economic markets function through the continuous interaction of supply and demand, mediated by price signals that coordinate the decisions of countless independent actors. When demand for a good exceeds available supply, prices tend to rise, which simultaneously discourages some consumption while incentivizing increased production. This self-correcting mechanism, first systematically described by classical economists, forms the theoretical foundation for understanding how decentralized economic systems can achieve a rough equilibrium without centralized planning, though real markets are complicated by externalities, information asymmetries, and market power.",
    "The human brain contains approximately eighty-six billion neurons, each capable of forming thousands of synaptic connections with other neurons, creating a network of staggering complexity that underlies all cognition, perception, and behavior. Neural signals propagate through electrochemical processes, where an action potential travels down an axon and triggers the release of neurotransmitters at the synapse, which then bind to receptors on the receiving neuron and either excite or inhibit its activity. Learning and memory are thought to arise from changes in the strength of these synaptic connections over time.",
    "Ocean currents play a critical role in regulating global climate by transporting vast quantities of heat around the planet. The thermohaline circulation, sometimes called the global conveyor belt, moves water driven by differences in temperature and salinity, carrying warm surface water from the tropics toward the poles and returning cold, dense water along the ocean floor. This circulation pattern takes roughly a thousand years to complete a full cycle, meaning changes introduced today may not manifest their full effects for many generations.",
]


def topic_switch_prompts(switch_every: int = 3) -> list[str]:
    """Interleaves the three categories above in round-robin blocks of
    `switch_every` items, simulating within-session topic switches
    (IMPLEMENTATION_PLAN.md #2.3's adversarial workload).

    An earlier version tracked pool position with a single `pool_order`
    counter that both selected the pool (`pool_order % 3`) and, on hitting
    `switch_every`, jumped forward by an extra step. Once the two smaller
    pools (LONG_FORM_PROSE, MULTI_TURN_CHAT) were exhausted, that extra jump
    made `pool_order % 3` permanently skip index 0 (FOCUSED_CODE) --
    `len(out)` got stuck below `total` and `while len(out) < total:` spun
    forever. Found by an actual hung run (30+ real minutes against a real
    model before the cause was isolated), not by inspection -- this
    function was never executed before that. Rewritten to track each pool
    as a mutable queue and a count of pools still having items, which can
    only decrease, so the loop is structurally guaranteed to terminate.
    """
    pools = [list(FOCUSED_CODE), list(MULTI_TURN_CHAT), list(LONG_FORM_PROSE)]
    out: list[str] = []
    pool_idx = 0
    while any(pools):
        pool = pools[pool_idx % len(pools)]
        for _ in range(switch_every):
            if not pool:
                break
            out.append(pool.pop(0))
        pool_idx += 1
    return out
