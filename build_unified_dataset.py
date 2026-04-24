"""
Unified Dataset Builder v3 — Final Clean Version
==================================================
Maps NIC → ISIC + NAICS with manual curated mappings,
then synthesizes keyword-based training data.
"""

import pandas as pd
import numpy as np
import json, os, random
from collections import defaultdict

random.seed(42)
os.makedirs("data", exist_ok=True)

# =====================================================
# 1. LOAD SOURCE DATA
# =====================================================

print("Loading datasets...")
nic_df = pd.read_csv("data/industries.csv")
nic_df["nic_code"] = nic_df["nic_code"].astype(str)
print(f"  NIC: {len(nic_df)} rows, {nic_df['nic_code'].nunique()} unique codes")

isic_df = pd.read_excel("data/ISIC_Unified.xlsx", sheet_name="ISIC_Rev4")
isic_df = isic_df[["ISIC Code", "Description"]].dropna()
isic_df.columns = ["code", "label"]
isic_df["code"] = isic_df["code"].astype(str)
ISIC_LOOKUP = dict(zip(isic_df["code"], isic_df["label"]))
print(f"  ISIC: {len(isic_df)} codes")

naics_df = pd.read_csv("data/codes_naics.csv", dtype=str).dropna()
NAICS_LOOKUP = dict(zip(naics_df["Code"], naics_df["Name"]))
valid_naics_6 = {c for c in naics_df["Code"] if len(c) == 6}
print(f"  NAICS: {len(valid_naics_6)} codes (6-digit)")

# =====================================================
# 2. COMPLETE MANUAL CROSSWALK — all 212 NIC codes
#    ISIC: verified against ISIC_Rev4 sheet
#    NAICS: verified against codes_naics.csv (Canadian NAICS)
# =====================================================

# Format: NIC_CODE: (ISIC_CODE, NAICS_CODE)
CROSSWALK = {
    # --- AGRICULTURE ---
    "1111":  ("A0111", "111140"),  # Wheat
    "1121":  ("A0112", "111160"),  # Basmati rice
    "1122":  ("A0112", "111160"),  # Non-basmati rice
    "1131":  ("A0113", "111219"),  # Vegetables
    "1132":  ("A0113", "111219"),  # Fruit-bearing vegetables
    "1133":  ("A0113", "111219"),  # Onion
    "1135":  ("A0113", "111211"),  # Potatoes
    "1140":  ("A0114", "111930"),  # Sugarcane → Sugar cane farming
    "1161":  ("A0116", "111920"),  # Cotton
    "1162":  ("A0116", "111940"),  # Jute/fibre
    "1221":  ("A0122", "111330"),  # Mangoes → Non-citrus fruit farming
    "1222":  ("A0122", "111330"),  # Bananas
    "1261":  ("A0126", "111419"),  # Coconut → Other food crops
    "1271":  ("A0127", "111419"),  # Tea → Other food crops
    "1272":  ("A0127", "111419"),  # Coffee
    "1281":  ("A0128", "111419"),  # Turmeric
    "1282":  ("A0128", "111419"),  # Chillies
    "1284":  ("A0128", "111419"),  # Betel
    "1412":  ("A0141", "112120"),  # Dairy cattle
    "1441":  ("A0144", "112410"),  # Sheep/goats
    "1461":  ("A0146", "112320"),  # Poultry broiler
    "1462":  ("A0146", "112310"),  # Egg production
    "1492":  ("A0149", "112910"),  # Beekeeping → Apiculture
    "3111":  ("A0311", "114113"),  # Ocean fishing
    "3221":  ("A0321", "112510"),  # Freshwater fish farming → Aquaculture

    # --- MINING ---
    "5101":  ("B0510", "212114"),  # Coal mining
    "7100":  ("B0710", "212210"),  # Iron ore
    "8101":  ("B0810", "212316"),  # Marble quarrying → non-metallic mineral
    "8102":  ("B0810", "212316"),  # Granite quarrying
    "8106":  ("B0810", "212323"),  # Sand/gravel
    "8107":  ("B0810", "212315"),  # Limestone

    # --- MANUFACTURING ---
    "10201": ("C1020", "311710"),  # Fish processing → Seafood product prep
    "10204": ("C1020", "311710"),  # Fish preserving
    "10304": ("C1030", "311410"),  # Fruit juices → Frozen food
    "10305": ("C1030", "311420"),  # Sauces/jams → Fruit/veg canning
    "10306": ("C1030", "311420"),  # Pickles
    "10504": ("C1050", "311515"),  # Butter/cheese/ghee
    "10505": ("C1050", "311520"),  # Ice cream
    "10502": ("C1050", "311511"),  # Milk powder → Fluid milk
    "10611": ("C1061", "311211"),  # Flour milling
    "10612": ("C1061", "311214"),  # Rice milling
    "10613": ("C1061", "311211"),  # Dal milling → Flour milling
    "10711": ("C1071", "311814"),  # Bread → Commercial bakeries
    "10712": ("C1071", "311814"),  # Biscuits/cakes
    "10721": ("C1072", "311310"),  # Sugar refining → Sugar manufacturing
    "10722": ("C1072", "311310"),  # Gur from sugarcane
    "10732": ("C1073", "311340"),  # Chocolate → Non-chocolate confectionery
    "10734": ("C1079", "311340"),  # Sweetmeats
    "10740": ("C1074", "311823"),  # Macaroni/noodles → Dry pasta/dough
    "10791": ("C1079", "311920"),  # Tea processing → Coffee/tea mfg
    "10795": ("C1079", "311940"),  # Spices
    "10796": ("C1079", "311919"),  # Papads → Other snack food
    "10801": ("C1080", "311119"),  # Animal feed → Other animal food
    "11041": ("C1104", "312110"),  # Aerated drinks → Soft drink mfg
    "11043": ("C1104", "312110"),  # Mineral water
    "11045": ("C1104", "312110"),  # Soft drinks
    "13111": ("C1311", "313110"),  # Cotton spinning
    "13121": ("C1312", "313210"),  # Cotton weaving
    "13122": ("C1312", "313210"),  # Silk weaving
    "13913": ("C1391", "313240"),  # Knitted synthetics → Knit fabric mills
    "13931": ("C1393", "314110"),  # Carpets → Carpet mills
    "13942": ("C1394", "314990"),  # Cordage/rope → Other textile products
    "13991": ("C1399", "313220"),  # Embroidery → Narrow fabric mills
    "13992": ("C1399", "313220"),  # Zari work
    "14101": ("C1410", "315190"),  # Textile garments → Other clothing
    "14105": ("C1410", "315190"),  # Custom tailoring
    "15121": ("C1512", "316990"),  # Travel goods → Other leather products
    "15201": ("C1520", "316210"),  # Footwear
    "16211": ("C1621", "321211"),  # Plywood/veneer
    "17023": ("C1702", "322219"),  # Cardboard boxes → Other paperboard container
    "17024": ("C1702", "322220"),  # Paper bags → Paper bag manufacturing
    "18112": ("C1811", "323119"),  # Printing → Other printing
    "18119": ("C1811", "323119"),  # Other printing
    "20121": ("C2012", "325314"),  # Fertilizers → Mixed fertilizer mfg
    "20212": ("C2029", "325610"),  # Disinfectants → Soap/cleaning compounds
    "20221": ("C2022", "325510"),  # Paints/varnishes
    "20231": ("C2023", "325610"),  # Soap → Soap/cleaning compounds
    "20233": ("C2023", "325610"),  # Detergent
    "20234": ("C2023", "325620"),  # Perfumes → Toilet preparation
    "20235": ("C2023", "325620"),  # Deodorants
    "20236": ("C2023", "325620"),  # Hair oil/shampoo
    "20237": ("C2023", "325620"),  # Cosmetics
    "20238": ("C2029", "325999"),  # Agarbatti → All other misc chemical products
    "20291": ("C2029", "325999"),  # Matches
    "21002": ("C2100", "325410"),  # Allopathic pharma
    "21003": ("C2100", "325410"),  # Ayurvedic pharma
    "21004": ("C2100", "325410"),  # Homeopathic pharma
    "21006": ("C2100", "339110"),  # Medical wadding/gauze
    "22111": ("C2211", "326210"),  # Rubber tyres
    "22191": ("C2219", "326290"),  # Rubber plates/tubes
    "22201": ("C2220", "326198"),  # Plastic semi-finished → All other plastics
    "22202": ("C2220", "326198"),  # Plastic household
    "22203": ("C2220", "326198"),  # Plastic packaging
    "23106": ("C2310", "327214"),  # Glass bangles → Glass
    "23921": ("C2392", "327120"),  # Clay bricks → Clay building material
    "23931": ("C2393", "327110"),  # Porcelain/pottery
    "23941": ("C2394", "327310"),  # Cement
    "24101": ("C2410", "331110"),  # Pig iron → Iron/steel mills
    "24103": ("C2410", "331110"),  # Steel
    "24201": ("C2420", "331420"),  # Copper products
    "24202": ("C2420", "331317"),  # Aluminium products
    "25931": ("C2593", "332210"),  # Cutlery → Cutlery/hand tool
    "25934": ("C2593", "332510"),  # Padlocks/locks → Hardware
    "25991": ("C2599", "332329"),  # Metal fasteners → Other metal stamping
    "25992": ("C2599", "332999"),  # Metal trunks → Misc metal products
    "25993": ("C2599", "331222"),  # Metal cable/wire → Steel wire drawing
    "25994": ("C2599", "332999"),  # Metal household articles
    "26101": ("C2610", "334410"),  # Electronic components → Semiconductor
    "26201": ("C2620", "334110"),  # Computers
    "26305": ("C2630", "334210"),  # Cellphones → Telephone equipment
    "27201": ("C2720", "335910"),  # Batteries → Battery manufacturing
    "27320": ("C2731", "331222"),  # Wires/cables → Steel wire drawing
    "27331": ("C2790", "335315"),  # Switch/socket → Switchgear
    "27400": ("C2740", "335110"),  # Electric lighting → Electric bulb
    "27503": ("C2750", "335210"),  # Electric fans → Small appliance
    "28211": ("C2821", "333110"),  # Tractors → Agricultural implement
    "29301": ("C2930", "336390"),  # Motor vehicle parts
    "30911": ("C3091", "336120"),  # Motorcycles → Heavy duty truck mfg
    "30921": ("C3092", "336990"),  # Bicycles → Other transportation equip mfg
    "31001": ("C3100", "337110"),  # Wood furniture → Wood kitchen cabinet
    "32111": ("C3211", "448310"),  # Gold/silver jewellery → Jewellery stores
    "32112": ("C3211", "448310"),  # Diamonds/precious stones
    "32120": ("C3212", "448310"),  # Imitation jewellery
    "32300": ("C3230", "451110"),  # Sports goods → Sporting goods stores
    "32401": ("C3240", "451120"),  # Dolls/toys → Hobby/toy/game stores
    "32504": ("C3250", "339110"),  # Medical instruments
    "33121": ("C3312", "811310"),  # Engine repair → Commercial machinery repair

    # --- ENERGY ---
    "35105": ("D3510", "221119"),  # Solar power → Other electric power gen
    "35106": ("D3510", "221119"),  # Wind power
    "35109": ("D3510", "221122"),  # Electric distribution
    "35202": ("D3520", "221210"),  # Gas distribution

    # --- CONSTRUCTION ---
    "41001": ("F4100", "236220"),  # Building construction → Commercial building
    "42101": ("F4210", "237310"),  # Highways/roads
    "43211": ("F4321", "238210"),  # Electrical installation
    "43221": ("F4322", "238220"),  # Plumbing/HVAC
    "43303": ("F4330", "238320"),  # Painting/decorating

    # --- TRADE ---
    "45200": ("G4520", "441310"),  # Motor vehicle repair → Auto parts stores
    "45403": ("G4540", "441220"),  # Motorcycle maintenance → Motorcycle dealers
    "47190": ("G4719", "452910"),  # Non-specialized retail → Warehouse clubs
    "47211": ("G4721", "445110"),  # Cereals/pulses → Supermarkets
    "47212": ("G4721", "445230"),  # Fresh fruit/vegetables → Fruit/veg markets
    "47300": ("G4730", "447110"),  # Automotive fuel → Gasoline stations
    "47414": ("G4741", "443110"),  # Telecom equipment → Appliance/electronics stores
    "47420": ("G4741", "443110"),  # Audio/video equipment
    "47522": ("G4752", "444130"),  # Hardware → Hardware stores
    "47591": ("G4759", "442110"),  # Household furniture → Furniture stores
    "47613": ("G4761", "451210"),  # Stationery → Book stores
    "47711": ("G4771", "448140"),  # Readymade garments → Family clothing stores
    "47721": ("G4772", "446110"),  # Pharmaceuticals → Pharmacies
    "47733": ("G4773", "448310"),  # Jewellery retail → Jewellery stores
    "47912": ("G4791", "454111"),  # E-commerce → Internet shopping

    # --- TRANSPORT ---
    "49221": ("H4922", "485210"),  # Long distance bus → Interurban bus
    "49224": ("H4922", "485310"),  # Taxi → Taxi service
    "49231": ("H4923", "484110"),  # Road freight → General freight trucking
    "52101": ("H5210", "493120"),  # Refrigerated warehousing
    "52102": ("H5210", "493110"),  # Non-refrigerated warehousing → General warehousing
    "52291": ("N7911", "561510"),  # Travel agents → Travel agencies
    "53200": ("H5320", "492110"),  # Courier

    # --- HOSPITALITY ---
    "55101": ("I5510", "721111"),  # Hotels → Hotels
    "56101": ("I5610", "722210"),  # Restaurants → Full-service restaurants
    "56102": ("I5610", "722210"),  # Fast food → Full-service restaurants
    "56210": ("I5621", "722310"),  # Event catering → Food service contractors
    "56291": ("I5629", "722310"),  # Mess/canteen
    "56292": ("I5629", "722310"),  # Tiffin services
    "56302": ("I5630", "722410"),  # Tea stalls → Drinking places
    "56303": ("I5630", "722410"),  # Juice bars

    # --- IT SERVICES ---
    "61103": ("J6120", "515210"),  # Cable operators → Pay/specialty TV
    "61104": ("J6110", "517111"),  # Internet access → Wired telecom carriers
    "62011": ("J6201", "541510"),  # Software → Computer systems design
    "62012": ("J6201", "541510"),  # Web design
    "62020": ("J6202", "541510"),  # IT consulting
    "63114": ("J6311", "518210"),  # Data entry → Data processing
    "63992": ("J6312", "519190"),  # Cyber cafe → Other info services

    # --- FINANCE ---
    "64910": ("K6491", "522291"),  # Microfinance → Consumer lending
    "64920": ("K6492", "522299"),  # Chit funds → Other non-depository credit
    "66120": ("K6612", "523120"),  # Securities dealing
    "66220": ("K6622", "524210"),  # Insurance agents → Insurance agencies
    "69201": ("M6920", "541212"),  # Accounting → Offices of accountants
    "69202": ("M6920", "541213"),  # Tax consultancy

    # --- REAL ESTATE ---
    "68200": ("L6820", "531211"),  # Real estate on fee basis → Real estate agents

    # --- PROFESSIONAL ---
    "69100": ("M6910", "541110"),  # Legal → Offices of lawyers
    "71100": ("M7110", "541310"),  # Architecture/engineering
    "73100": ("M7310", "541810"),  # Advertising
    "74101": ("M7410", "541490"),  # Fashion design → Other specialized design
    "74103": ("M7410", "541430"),  # Graphic design
    "74201": ("M7420", "541920"),  # Photography → Photographic services
    "75000": ("M7500", "541940"),  # Veterinary

    # --- EDUCATION ---
    "85211": ("P8521", "611110"),  # Primary school → Elementary school
    "85221": ("P8522", "611510"),  # Vocational education → Technical/trade schools
    "85223": ("P8549", "611690"),  # Driving school → Other schools
    "85301": ("P8530", "611310"),  # Higher education → Universities
    "85420": ("P8542", "611610"),  # Cultural education → Fine arts schools
    "85491": ("P8549", "611690"),  # Academic tutoring → Other schools

    # --- HEALTHCARE ---
    "86100": ("Q8610", "622111"),  # Hospital → General hospitals
    "86201": ("Q8620", "621110"),  # Medical practice → Offices of physicians
    "86202": ("Q8620", "621210"),  # Dental practice → Offices of dentists
    "86901": ("Q8690", "621390"),  # Ayurveda → Other health practitioners
    "86903": ("Q8690", "621390"),  # Homeopaths
    "86905": ("Q8690", "621510"),  # Diagnostic labs → Medical labs

    # --- SERVICES ---
    "78100": ("N7810", "561310"),  # Employment agencies
    "80100": ("N8010", "561612"),  # Private security → Security guard services
    "81210": ("N8121", "561722"),  # Building cleaning → Janitorial services
    "81299": ("N8129", "561722"),  # Other cleaning → Janitorial services
    "82191": ("N8219", "561430"),  # Photocopying → Business service centres
    "82200": ("N8220", "561420"),  # Call centres → Telephone call centres
    "82300": ("N8230", "561920"),  # Conventions/trade shows → Convention organizers
    "95120": ("S9512", "811210"),  # Communication equip repair → Electronic repair
    "95210": ("S9521", "811210"),  # Consumer electronics repair
    "95230": ("S9523", "811430"),  # Footwear repair → Reupholstery/furniture repair
    "95291": ("S9529", "811490"),  # Bicycle repair → Other personal goods repair
    "96010": ("S9601", "812320"),  # Laundry/dry cleaning
    "96020": ("S9602", "812115"),  # Hair/beauty → Hairdressing salons
}

# =====================================================
# 3. BUILD CROSSWALK JSON
# =====================================================

print("\nBuilding crosswalk...")
nic_unique = nic_df[["nic_code", "nic_label", "division"]].drop_duplicates(subset=["nic_code"])

crosswalk = {}
missing_isic = []
missing_naics = []

for _, row in nic_unique.iterrows():
    nic = str(row["nic_code"])
    mapping = CROSSWALK.get(nic)
    
    if mapping is None:
        print(f"  WARNING: NIC {nic} ({row['nic_label']}) has no manual mapping!")
        continue
    
    isic_code, naics_code = mapping
    isic_label = ISIC_LOOKUP.get(isic_code, "—")
    naics_label = NAICS_LOOKUP.get(naics_code, "—")
    
    if isic_label == "—":
        missing_isic.append((nic, isic_code))
    if naics_label == "—":
        missing_naics.append((nic, naics_code))
    
    crosswalk[nic] = {
        "nic_label": row["nic_label"], "division": row["division"],
        "isic_code": isic_code, "isic_label": isic_label,
        "naics_code": naics_code, "naics_label": naics_label,
    }

if missing_isic:
    print(f"  ⚠ {len(missing_isic)} ISIC codes not found in dataset")
if missing_naics:
    print(f"  ⚠ {len(missing_naics)} NAICS codes not found in dataset:")
    for nic, nc in missing_naics[:10]:
        print(f"    NIC {nic} → NAICS {nc}")

with open("data/code_crosswalk.json", "w", encoding="utf-8") as f:
    json.dump(crosswalk, f, indent=2, ensure_ascii=False)
print(f"  Crosswalk: {len(crosswalk)} entries saved → data/code_crosswalk.json")

# =====================================================
# 4. ADD ISIC/NAICS COLUMNS TO ORIGINAL DATASET
# =====================================================

for col in ["isic_code", "isic_label", "naics_code", "naics_label"]:
    nic_df[col] = nic_df["nic_code"].map(lambda x, c=col: crosswalk.get(x, {}).get(c, "UNKNOWN"))

# =====================================================
# 5. KEYWORD-BASED SYNTHESIS (conflict-free)
# =====================================================

STOP = {
    "of","and","the","in","for","a","an","on","or","to","by","at","with",
    "from","as","is","are","was","be","not","all","other","etc","including",
    "activities","services","manufacture","manufacturing","growing",
    "production","processing","preparation","sale","retail","wholesale",
    "repair","maintenance","general","operation","specialized","elsewhere",
    "stores","articles","products","equipment","materials","goods","items",
    "agents","brokers","agencies","related","forms","types","kind","short",
    "term","long","made","based","used","using","various","similar","like",
    "non-specialized","non","except",
}

TEMPLATES = [
    "{kw}", "{kw} business", "{kw} shop", "{kw} unit",
    "i sell {kw}", "i make {kw}", "we produce {kw}",
    "i deal in {kw}", "{kw} manufacturing", "{kw} trading",
    "{kw} store", "my business is {kw}", "i run a {kw} business",
    "we are into {kw}", "{kw} industry", "{kw} supplier",
    "small scale {kw}", "{kw} work", "{kw} enterprise",
    "home based {kw}", "{kw} in india", "msme {kw}",
]

def get_keywords(text):
    words = text.lower().replace(",", " ").replace("(", " ").replace(")", " ").replace(";", " ").split()
    words = [w.strip() for w in words if len(w) > 2 and w not in STOP]
    kws = set(words)
    for i in range(len(words) - 1):
        kws.add(f"{words[i]} {words[i+1]}")
    if len(words) >= 2:
        kws.add(" ".join(words))
    return kws

# Collect keywords per NIC code
print("\nExtracting keywords...")
nic_kws = {}
for nic, info in crosswalk.items():
    kws = set()
    for src in [info["nic_label"], info["isic_label"], info["naics_label"]]:
        if src != "—":
            kws.update(get_keywords(src))
    nic_kws[nic] = kws

# Remove conflicting keywords (same kw → multiple NIC codes)
kw_owners = defaultdict(set)
for nic, kws in nic_kws.items():
    for kw in kws:
        kw_owners[kw].add(nic)

conflicts = {kw for kw, owners in kw_owners.items() if len(owners) > 1}
print(f"  Keywords: {len(kw_owners)} total, {len(conflicts)} conflicting → removed")

# Generate conflict-free synthetic rows
synth_rows = []
seen = set()

for nic, info in crosswalk.items():
    safe_kws = nic_kws[nic] - conflicts
    for kw in safe_kws:
        for tmpl in random.sample(TEMPLATES, min(6, len(TEMPLATES))):
            text = tmpl.format(kw=kw).strip()
            if text.lower() not in seen and len(text) > 2:
                seen.add(text.lower())
                synth_rows.append({
                    "text": text, "nic_code": nic, "nic_label": info["nic_label"],
                    "division": info["division"], "isic_code": info["isic_code"],
                    "isic_label": info["isic_label"], "naics_code": info["naics_code"],
                    "naics_label": info["naics_label"],
                })

synth_df = pd.DataFrame(synth_rows)

# Remove any that duplicate existing base data
existing = set(nic_df["text"].str.lower())
synth_df = synth_df[~synth_df["text"].str.lower().isin(existing)]
print(f"  Synthesized: {len(synth_df)} conflict-free rows")

# Combine and deduplicate
unified = pd.concat([nic_df, synth_df], ignore_index=True)
unified = unified.drop_duplicates(subset=["text", "nic_code"])

# Final conflict check
text_codes = unified.groupby("text")["nic_code"].nunique()
bad_texts = set(text_codes[text_codes > 1].index)
if bad_texts:
    print(f"  Removing {len(bad_texts)} remaining text conflicts...")
    unified = unified[~unified["text"].isin(bad_texts)]
else:
    print(f"  ✓ Zero text conflicts")

unified = unified.sample(frac=1, random_state=42).reset_index(drop=True)

# =====================================================
# 6. SAVE
# =====================================================

unified.to_csv("data/industries_unified.csv", index=False, encoding="utf-8")

print(f"\n{'='*60}")
print(f"  FINAL DATASET STATS")
print(f"  Original rows     : {len(nic_df)}")
print(f"  Synthesized rows  : {len(synth_df)}")
print(f"  Total samples     : {len(unified)}")
print(f"  Unique NIC codes  : {unified['nic_code'].nunique()}")
print(f"  Unique ISIC codes : {unified['isic_code'].nunique()}")
print(f"  Unique NAICS codes: {unified['naics_code'].nunique()}")
print(f"{'='*60}")

# Spot-check
checks = ["1140","10505","20291","85420","55101","47414","21002","56101","86100","32111"]
print(f"\n  Spot-check:")
for nic in checks:
    i = crosswalk.get(nic, {})
    il = i.get("isic_label","?")[:30]
    nl = i.get("naics_label","?")[:30]
    print(f"  {nic:<6} → ISIC {i.get('isic_code','?'):<6} ({il}) | NAICS {i.get('naics_code','?'):<6} ({nl})")
