"""
Ambiguous Patterns — Udyam NIC Clarifier
==========================================
Covers English + Hindi + Hinglish triggers.
Each pattern fires when confidence < CONFIDENCE_THRESHOLD.
Only ONE question is asked — the first matching pattern wins.
Order matters: more specific patterns go first.
"""

CONFIDENCE_THRESHOLD = 0.45

AMBIGUOUS_PATTERNS = [

    # ----------------------------------------------------------------
    # REPAIR / MAINTENANCE
    # ----------------------------------------------------------------
    {
        "keywords": [
            "repair", "fix", "mend", "servicing", "maintain",
            "maintenance", "workshop", "garage",
            "sudharna", "theek karna", "theek", "repairing", "mistri", "mechanic",
        ],
        "question": "What do you repair? Choose the closest:",
        "options": [
            "vehicles / cars / bikes / trucks",
            "mobile phones / computers / electronics",
            "industrial machines / generators / pumps",
            "clothes / shoes / furniture / household items",
            "buildings / plumbing / electrical wiring",
        ],
    },

    # ----------------------------------------------------------------
    # SELLING / TRADING
    # ----------------------------------------------------------------
    {
        "keywords": [
            "sell", "selling", "shop", "store", "retail", "trader",
            "trading", "dealer", "dealership", "supply", "supplier",
            "wholesale", "wholesaler", "distributor", "stockist",
            "merchant", "vendor",
            "dukan", "becna", "bechna", "bechta", "bikri", "vyapar",
            "vyapari", "thela", "mandi", "kiryana",
        ],
        "question": "What do you mainly sell or trade?",
        "options": [
            "food / groceries / vegetables / dairy",
            "clothes / garments / fabrics / footwear",
            "medicines / medical supplies",
            "electronics / mobiles / computers",
            "hardware / tools / construction material",
            "vehicles / vehicle parts / fuel",
            "jewellery / gold / silver",
            "agricultural inputs (seeds, fertilisers, pesticides)",
            "other goods",
        ],
    },

    # ----------------------------------------------------------------
    # MAKING / MANUFACTURING
    # ----------------------------------------------------------------
    {
        "keywords": [
            "make", "making", "manufacture", "manufacturing", "produce",
            "producing", "production", "fabricate", "fabrication",
            "assemble", "assembly", "craft", "crafting", "unit",
            "factory", "plant",
            "banane", "banata", "banate", "bana", "taiyar karna",
            "nirmaan", "udyog", "karkhana",
        ],
        "question": "What product do you make or manufacture?",
        "options": [
            "food products (flour, sweets, pickles, spices, oil)",
            "garments / textiles / bags / shoes",
            "soap / detergent / cosmetics / agarbatti / candle",
            "metal goods / utensils / tools / locks / jewellery",
            "furniture / wood products / plywood",
            "plastic / rubber products",
            "bricks / tiles / cement / pottery / glass",
            "medicines / ayurvedic / herbal products",
            "electronics / wires / bulbs / batteries",
            "other (type the product name)",
        ],
    },

    # ----------------------------------------------------------------
    # FARMING / GROWING
    # ----------------------------------------------------------------
    {
        "keywords": [
            "farm", "farming", "grow", "growing", "cultivate",
            "cultivation", "crop", "crops", "harvest", "field",
            "agriculture", "agricultural", "kisan", "plantation", "orchard",
            "kheti", "khet", "fasal", "ugana", "bagicha", "palan", "bagwani",
        ],
        "question": "What do you farm or grow?",
        "options": [
            "cereal crops (wheat, rice, maize, millets)",
            "vegetables (onion, tomato, potato, leafy vegetables)",
            "fruits (mango, banana, citrus, grapes)",
            "cash crops (cotton, sugarcane, jute, tobacco)",
            "spices / herbs (chili, ginger, turmeric, cardamom)",
            "livestock / dairy (cow, buffalo, goat, poultry)",
            "fish / prawn / aquaculture",
            "horticulture / flowers / nursery",
        ],
    },

    # ----------------------------------------------------------------
    # CONSTRUCTION
    # ----------------------------------------------------------------
    {
        "keywords": [
            "construction", "construct", "build", "building", "builder",
            "contractor", "civil", "infrastructure", "erect",
            "develop", "developer",
            "nirmaan", "nirman", "thekedaar", "thekedar", "makan",
        ],
        "question": "What type of construction work do you do?",
        "options": [
            "residential buildings / houses / flats",
            "commercial buildings / offices / shops",
            "roads / highways / bridges / tunnels",
            "electrical installation / wiring",
            "plumbing / sanitation work",
            "interior decoration / painting / tiling",
            "industrial / factory construction",
        ],
    },

    # ----------------------------------------------------------------
    # SERVICES — broad bucket
    # ----------------------------------------------------------------
    {
        "keywords": [
            "service", "services", "provide", "providing", "offer",
            "offering", "agency", "consultant", "consulting", "security",
            "center",
            "seva", "suvidha", "karya",
        ],
        "question": "What kind of service do you provide?",
        "options": [
            "IT / software / web / app development",
            "accounting / tax / legal / financial",
            "beauty / salon / spa / laundry",
            "security / housekeeping / pest control",
            "education / coaching / training",
            "healthcare / medical / diagnostic",
            "event management / catering",
            "transport / courier / logistics",
            "repair of electronics / appliances / vehicles",
            "advertising / marketing / photography / design",
        ],
    },

    # ----------------------------------------------------------------
    # TRANSPORT / LOGISTICS
    # ----------------------------------------------------------------
    {
        "keywords": [
            "transport", "transportation", "logistic", "logistics",
            "delivery", "courier", "cargo", "freight", "move",
            "moving", "haul", "haulage", "fleet",
            "parivahan", "dhulai", "maal", "gaadi",
        ],
        "question": "What type of transport or logistics do you do?",
        "options": [
            "goods transport by truck / tempo",
            "passenger transport (taxi / auto / cab / bus)",
            "courier / parcel delivery",
            "warehousing / cold storage",
            "shipping / freight forwarding",
        ],
    },

    # ----------------------------------------------------------------
    # FOOD — manufacturing vs hospitality vs retail
    # ----------------------------------------------------------------
    {
        "keywords": [
            "food", "khana", "khaana", "catering", "tiffin",
            "cooking", "cook", "bakery", "confectionery",
            "restaurant", "dhaba", "canteen", "hotel",
            "snack", "snacks", "mithai", "sweet",
        ],
        "question": "Is this about making packaged food or serving food to customers?",
        "options": [
            "making packaged food products (biscuits, pickles, masala, ghee)",
            "running a restaurant / dhaba / hotel / canteen",
            "catering service (events / offices / tiffin delivery)",
            "bakery (making breads, cakes, biscuits to sell)",
            "tea / juice / snack stall",
        ],
    },

    # ----------------------------------------------------------------
    # PROCESSING / MILLING
    # ----------------------------------------------------------------
    {
        "keywords": [
            "process", "processing", "mill", "milling", "grind",
            "grinding", "pack", "packing", "packaging",
            "chakki", "pisna", "peesna",
        ],
        "question": "What material do you process or mill?",
        "options": [
            "grains (wheat flour / rice / dal / maize)",
            "spices / masala",
            "oil seeds (groundnut / mustard / sesame)",
            "tea / coffee",
            "fish / meat / poultry",
            "fruits / vegetables (juice, puree, dried)",
        ],
    },

    # ----------------------------------------------------------------
    # PRINTING / PUBLISHING
    # ----------------------------------------------------------------
    {
        "keywords": [
            "print", "printing", "press", "publish", "publishing",
            "design", "designing",
        ],
        "question": "What is your printing or design work?",
        "options": [
            "printing press (books, magazines, newspapers)",
            "packaging / label / sticker printing",
            "graphic design / logo / branding",
            "textile / fabric printing",
            "digital / offset printing shop",
        ],
    },

    # ----------------------------------------------------------------
    # HEALTH / MEDICAL
    # ----------------------------------------------------------------
    {
        "keywords": [
            "health", "medical", "medicine", "clinic", "doctor",
            "hospital", "pharma", "pharmaceutical", "ayurved",
            "ayurvedic", "herbal", "chikitsa", "dawai", "dawa",
        ],
        "question": "What is your health or medical activity?",
        "options": [
            "running a clinic / hospital / nursing home",
            "manufacturing medicines / tablets / ayurvedic products",
            "selling medicines (pharmacy / medical store)",
            "diagnostic / pathology lab",
            "home healthcare / nursing / physiotherapy",
        ],
    },

    # ----------------------------------------------------------------
    # EDUCATION / TRAINING
    # ----------------------------------------------------------------
    {
        "keywords": [
            "education", "school", "college", "teach", "teaching",
            "tuition", "coaching", "training", "institute", "academy",
            "class", "classes", "vidyalaya", "padhana", "padhna",
            "siksha", "shiksha",
        ],
        "question": "What kind of education or training do you provide?",
        "options": [
            "school (primary / secondary)",
            "coaching / tuition centre",
            "skill / vocational training (ITI, diploma, trades)",
            "driving school",
            "dance / music / arts classes",
            "computer / IT training",
        ],
    },

    # ----------------------------------------------------------------
    # ENERGY / POWER
    # ----------------------------------------------------------------
    {
        "keywords": [
            "solar", "energy", "power", "electricity", "wind",
            "renewable", "gas", "lpg", "cylinder",
            "urja", "bijli", "gas agency",
        ],
        "question": "What is your energy business?",
        "options": [
            "solar panel installation / rooftop solar",
            "solar panel manufacturing",
            "wind energy / mini hydro power",
            "LPG / CNG gas agency / distributor",
            "electric power distribution",
        ],
    },

    # ----------------------------------------------------------------
    # BEAUTY / PERSONAL CARE
    # ----------------------------------------------------------------
    {
        "keywords": [
            "beauty", "salon", "parlour", "parlor", "spa", "hair",
            "cosmetic", "makeup", "mehendi", "mehandi", "nails",
            "grooming", "barber",
            "nai", "sundarta",
        ],
        "question": "What is your beauty or personal care business?",
        "options": [
            "beauty parlour / salon / spa (services to customers)",
            "manufacturing cosmetics / beauty products",
            "selling cosmetics (retail shop)",
        ],
    },

    # ----------------------------------------------------------------
    # IT / DIGITAL
    # ----------------------------------------------------------------
    {
        "keywords": [
            "software", "app", "website", "web", "digital", "tech",
            "technology", "computer", "mobile", "cyber",
            "data", "cloud", "startup",
        ],
        "question": "What is your IT or technology activity?",
        "options": [
            "custom software / app / website development",
            "IT consulting / support / managed services",
            "data entry / BPO / back-office",
            "cyber cafe / internet services",
            "hardware / computer assembly / repair",
            "e-commerce / online selling platform",
        ],
    },

    # ----------------------------------------------------------------
    # JEWELLERY
    # ----------------------------------------------------------------
    {
        "keywords": [
            "jewellery", "jewelry", "gold", "silver", "diamond",
            "gems", "sona", "chandi", "heera", "gehna", "zevarat", "sone", "sonar",
            "artificial jewellery", "imitation",
        ],
        "question": "What is your jewellery business?",
        "options": [
            "manufacturing gold / silver jewellery",
            "diamond cutting and polishing",
            "manufacturing imitation / fashion jewellery",
            "retail jewellery shop",
            "wholesale trading of gems and jewellery",
        ],
    },

    # ----------------------------------------------------------------
    # TEXTILE / GARMENTS
    # ----------------------------------------------------------------
    {
        "keywords": [
            "textile", "fabric", "cloth", "garment", "garments",
            "weave", "weaving", "stitch", "stitching", "loom",
            "kapda", "kapde", "bunai", "silai", "kadhai",
            "embroidery", "knitting", "hosiery",
        ],
        "question": "Which part of the textile / garment chain are you in?",
        "options": [
            "spinning yarn (cotton / silk / synthetic)",
            "weaving fabric (handloom / powerloom)",
            "garment manufacturing (readymade clothes)",
            "tailoring / custom stitching",
            "embroidery / zari / zardozi work",
            "selling clothes / fabrics (retail / wholesale)",
        ],
    },


    # ----------------------------------------------------------------
    # CATCH-ALL Hindi kaam / work type
    # ----------------------------------------------------------------
    {
        "keywords": [
            "kaam", "mera kaam", "hamara kaam", "ka kaam",
            "work", "my work", "our work", "business", "occupation",
        ],
        "question": "What is the main activity of your business?",
        "options": [
            "manufacturing / making a product",
            "selling / trading goods",
            "providing a service (repair, beauty, education, etc.)",
            "farming / agriculture / animal rearing",
            "construction / building work",
            "food / restaurant / catering",
        ],
    },

]

# ----------------------------------------------------------------
# Quick test — run this file directly to check keyword coverage
# ----------------------------------------------------------------
if __name__ == "__main__":
    test_phrases = [
        "i make soap at home",
        "mera kaam painting ka hai",
        "we repair bikes",
        "hum kapde bechte hain",
        "solar panel lagana",
        "i run a beauty parlour",
        "bakery products banana",
        "fish farming freshwater",
        "road construction contractor",
        "i sell medicines",
        "hum software banate hain",
        "sona chandi ka kaam",
        "our business is catering",
        "i do printing work",
        "we provide security services",
        "tuition centre",
    ]

    def find_pattern(text):
        t = text.lower()
        for i, p in enumerate(AMBIGUOUS_PATTERNS):
            if any(kw in t for kw in p["keywords"]):
                return i, p["question"]
        return None, "no match"

    print(f"\n{'Input':<40} {'Pattern #':<12} {'Question'}")
    print("-" * 90)
    for phrase in test_phrases:
        idx, q = find_pattern(phrase)
        print(f"{phrase:<40} {'#'+str(idx) if idx is not None else 'NONE':<12} {q[:50]}")