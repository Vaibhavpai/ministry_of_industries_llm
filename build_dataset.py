"""
Dataset Builder v2 — Udyam NIC Code Predictor
==============================================
Expanded to 3000+ raw seed entries for better per-class coverage.
Augmentation now generates 12 variants per seed = ~36k+ total samples.
Division names are NORMALIZED to match UDYAM_GUIDANCE keys exactly.
"""

import csv, json, random
from collections import Counter

random.seed(42)

# ============================================================
# NORMALIZED DIVISION NAMES — must match UDYAM_GUIDANCE keys
# ============================================================
AG  = "Agriculture"
MFG = "Manufacturing"
CON = "Construction"
TRD = "Trade"
TRN = "Transport"
HOS = "Hospitality"
IT  = "IT Services"
FIN = "Finance"
RE  = "Real Estate"
PRO = "Professional"
EDU = "Education"
HC  = "Healthcare"
SVC = "Services"
ENE = "Energy"
MIN = "Mining"

# ============================================================
# RAW DATA: (description, nic_code, nic_label, division)
# Format: plain English descriptions real users would type
# ============================================================
RAW_DATA = [

    # ========================
    # AGRICULTURE
    # ========================
    ("i grow wheat on my farm", "1111", "Growing of wheat", AG),
    ("wheat farming business", "1111", "Growing of wheat", AG),
    ("we cultivate wheat", "1111", "Growing of wheat", AG),
    ("wheat cultivation in punjab", "1111", "Growing of wheat", AG),
    ("wheat crop production", "1111", "Growing of wheat", AG),
    ("farming wheat and selling", "1111", "Growing of wheat", AG),
    ("we are wheat farmers", "1111", "Growing of wheat", AG),
    ("my field grows wheat every season", "1111", "Growing of wheat", AG),
    ("wheat grower and supplier", "1111", "Growing of wheat", AG),
    ("rabi crop wheat production", "1111", "Growing of wheat", AG),

    ("i grow rice paddy", "1121", "Organic farming of basmati rice", AG),
    ("basmati rice farming", "1121", "Organic farming of basmati rice", AG),
    ("paddy cultivation", "1121", "Organic farming of basmati rice", AG),
    ("rice farming in irrigated fields", "1121", "Organic farming of basmati rice", AG),
    ("we produce basmati rice", "1121", "Organic farming of basmati rice", AG),
    ("organic rice cultivation", "1121", "Organic farming of basmati rice", AG),
    ("i grow non basmati rice", "1122", "Organic farming of non-basmati rice", AG),
    ("non basmati paddy farming", "1122", "Organic farming of non-basmati rice", AG),

    ("vegetable farming", "1131", "Growing of vegetables", AG),
    ("i grow vegetables", "1131", "Growing of vegetables", AG),
    ("leafy vegetable cultivation", "1131", "Growing of vegetables", AG),
    ("spinach and cabbage farming", "1131", "Growing of vegetables", AG),
    ("we grow green vegetables", "1131", "Growing of vegetables", AG),
    ("seasonal vegetable farming", "1131", "Growing of vegetables", AG),

    ("i grow onions and tomatoes", "1132", "Growing of fruit-bearing vegetables", AG),
    ("tomato farming", "1132", "Growing of fruit-bearing vegetables", AG),
    ("cucumber and brinjal cultivation", "1132", "Growing of fruit-bearing vegetables", AG),
    ("we grow tomatoes and chillies", "1132", "Growing of fruit-bearing vegetables", AG),

    ("we grow onion", "1133", "Growing of onion", AG),
    ("onion farming in nashik", "1133", "Growing of onion", AG),
    ("onion cultivation and selling", "1133", "Growing of onion", AG),
    ("i am an onion farmer", "1133", "Growing of onion", AG),
    ("kanda farming", "1133", "Growing of onion", AG),

    ("potato farming", "1135", "Growing of potatoes and tubers", AG),
    ("i grow potatoes", "1135", "Growing of potatoes and tubers", AG),
    ("aloo cultivation", "1135", "Growing of potatoes and tubers", AG),
    ("potato and sweet potato farming", "1135", "Growing of potatoes and tubers", AG),

    ("sugarcane farming", "1140", "Growing of sugar cane", AG),
    ("we grow sugarcane", "1140", "Growing of sugar cane", AG),
    ("ganna cultivation", "1140", "Growing of sugar cane", AG),
    ("sugarcane crop production", "1140", "Growing of sugar cane", AG),

    ("cotton farming", "1161", "Growing of cotton", AG),
    ("i grow cotton", "1161", "Growing of cotton", AG),
    ("kapas farming", "1161", "Growing of cotton", AG),
    ("cotton cultivation in vidarbha", "1161", "Growing of cotton", AG),

    ("jute farming", "1162", "Growing of jute", AG),
    ("jute cultivation in west bengal", "1162", "Growing of jute", AG),

    ("i grow mangoes", "1221", "Growing of mangoes", AG),
    ("mango orchard", "1221", "Growing of mangoes", AG),
    ("mango farming in konkan", "1221", "Growing of mangoes", AG),
    ("aam bagicha", "1221", "Growing of mangoes", AG),
    ("alphonso mango farming", "1221", "Growing of mangoes", AG),
    ("we have mango trees and sell fruit", "1221", "Growing of mangoes", AG),

    ("banana farming", "1222", "Growing of bananas", AG),
    ("we grow bananas", "1222", "Growing of bananas", AG),
    ("kela ki kheti", "1222", "Growing of bananas", AG),
    ("banana plantation", "1222", "Growing of bananas", AG),

    ("coconut farming", "1261", "Growing of coconut", AG),
    ("nariyal ki kheti", "1261", "Growing of coconut", AG),
    ("coconut plantation in kerala", "1261", "Growing of coconut", AG),

    ("tea plantation", "1271", "Growing of tea", AG),
    ("we grow tea", "1271", "Growing of tea", AG),
    ("chai bagicha in assam", "1271", "Growing of tea", AG),

    ("coffee plantation", "1272", "Growing of coffee", AG),
    ("we grow coffee in coorg", "1272", "Growing of coffee", AG),

    ("i grow ginger", "1281", "Growing of ginger", AG),
    ("adrak ki kheti", "1281", "Growing of ginger", AG),

    ("chili farming", "1282", "Growing of chili", AG),
    ("we grow red chillies", "1282", "Growing of chili", AG),
    ("mirchi ki kheti", "1282", "Growing of chili", AG),

    ("spice farming", "1284", "Growing of spices", AG),
    ("mixed spice cultivation", "1284", "Growing of spices", AG),
    ("pepper and cardamom farming", "1284", "Growing of spices", AG),

    ("i keep bees and sell honey", "1492", "Bee-keeping and production of honey", AG),
    ("honey production business", "1492", "Bee-keeping and production of honey", AG),
    ("apiary and honey selling", "1492", "Bee-keeping and production of honey", AG),
    ("madhumakhi palan", "1492", "Bee-keeping and production of honey", AG),
    ("we do beekeeping", "1492", "Bee-keeping and production of honey", AG),

    ("poultry farming", "1461", "Raising and breeding of poultry", AG),
    ("i raise chickens for meat", "1461", "Raising and breeding of poultry", AG),
    ("broiler poultry farm", "1461", "Raising and breeding of poultry", AG),
    ("murgi palan", "1461", "Raising and breeding of poultry", AG),
    ("we have a chicken farm", "1461", "Raising and breeding of poultry", AG),
    ("layer poultry farm", "1461", "Raising and breeding of poultry", AG),

    ("egg production farm", "1462", "Production of eggs", AG),
    ("we produce eggs and sell", "1462", "Production of eggs", AG),
    ("anda farm", "1462", "Production of eggs", AG),

    ("dairy farming", "1412", "Production of milk from cows or buffaloes", AG),
    ("i have a cow dairy", "1412", "Production of milk from cows or buffaloes", AG),
    ("milk production from cows", "1412", "Production of milk from cows or buffaloes", AG),
    ("bhains palan dudh ka kaam", "1412", "Production of milk from cows or buffaloes", AG),
    ("buffalo dairy farm", "1412", "Production of milk from cows or buffaloes", AG),
    ("we sell milk from our farm", "1412", "Production of milk from cows or buffaloes", AG),
    ("cattle dairy business", "1412", "Production of milk from cows or buffaloes", AG),

    ("goat farming", "1441", "Raising and breeding of sheep and goats", AG),
    ("we raise goats and sheep", "1441", "Raising and breeding of sheep and goats", AG),
    ("bakri palan", "1441", "Raising and breeding of sheep and goats", AG),
    ("goat and sheep breeding", "1441", "Raising and breeding of sheep and goats", AG),

    ("fish farming", "3221", "Fish farming in freshwater", AG),
    ("freshwater aquaculture", "3221", "Fish farming in freshwater", AG),
    ("prawn farming in pond", "3221", "Fish farming in freshwater", AG),
    ("machli palan", "3221", "Fish farming in freshwater", AG),
    ("we do fish farming in our pond", "3221", "Fish farming in freshwater", AG),

    ("i do fishing in the sea", "3111", "Fishing in ocean and coastal waters", AG),
    ("fishing boat business", "3111", "Fishing in ocean and coastal waters", AG),
    ("sea fishing", "3111", "Fishing in ocean and coastal waters", AG),

    # ========================
    # MINING
    # ========================
    ("coal mining", "5101", "Opencast mining of hard coal", MIN),
    ("we mine coal opencast", "5101", "Opencast mining of hard coal", MIN),
    ("iron ore mining", "7100", "Mining of iron ores", MIN),
    ("granite quarrying", "8102", "Quarrying of granite", MIN),
    ("we quarry granite", "8102", "Quarrying of granite", MIN),
    ("marble quarrying", "8101", "Quarrying of marble", MIN),
    ("limestone mining", "8107", "Mining of limestone", MIN),
    ("sand quarrying", "8106", "Operation of sand or gravel pits", MIN),
    ("gravel pit operation", "8106", "Operation of sand or gravel pits", MIN),

    # ========================
    # FOOD MANUFACTURING
    # ========================
    ("rice mill", "10612", "Rice milling", MFG),
    ("i run a rice mill", "10612", "Rice milling", MFG),
    ("rice milling unit", "10612", "Rice milling", MFG),
    ("we mill rice and sell", "10612", "Rice milling", MFG),
    ("chawal chakki", "10612", "Rice milling", MFG),
    ("rice processing plant", "10612", "Rice milling", MFG),

    ("flour mill", "10611", "Flour milling", MFG),
    ("we grind flour", "10611", "Flour milling", MFG),
    ("atta chakki", "10611", "Flour milling", MFG),
    ("wheat flour milling", "10611", "Flour milling", MFG),
    ("chakki flour mill business", "10611", "Flour milling", MFG),
    ("we have a flour grinding unit", "10611", "Flour milling", MFG),

    ("dal mill", "10613", "Dal (pulses) milling", MFG),
    ("we mill dal and pulses", "10613", "Dal (pulses) milling", MFG),
    ("pulse milling unit", "10613", "Dal (pulses) milling", MFG),

    ("bakery products", "10712", "Manufacture of biscuits and cakes", MFG),
    ("we make bread and biscuits", "10712", "Manufacture of biscuits and cakes", MFG),
    ("biscuit manufacturing", "10712", "Manufacture of biscuits and cakes", MFG),
    ("cake and pastry making", "10712", "Manufacture of biscuits and cakes", MFG),
    ("bakery manufacturing unit", "10712", "Manufacture of biscuits and cakes", MFG),

    ("i make bread", "10711", "Manufacture of bread", MFG),
    ("bread bakery unit", "10711", "Manufacture of bread", MFG),
    ("bread manufacturing", "10711", "Manufacture of bread", MFG),
    ("we bake and sell bread", "10711", "Manufacture of bread", MFG),

    ("sugar manufacturing", "10721", "Manufacture of sugar from sugarcane", MFG),
    ("sugarcane sugar factory", "10721", "Manufacture of sugar from sugarcane", MFG),

    ("we make jaggery gur", "10722", "Manufacture of gur from sugarcane", MFG),
    ("jaggery making unit", "10722", "Manufacture of gur from sugarcane", MFG),
    ("gur production", "10722", "Manufacture of gur from sugarcane", MFG),
    ("i make gud from sugarcane", "10722", "Manufacture of gur from sugarcane", MFG),

    ("chocolate making", "10732", "Manufacture of chocolate", MFG),
    ("we manufacture chocolates", "10732", "Manufacture of chocolate", MFG),
    ("chocolate production unit", "10732", "Manufacture of chocolate", MFG),

    ("sweet shop making mithai", "10734", "Manufacture of sweetmeats", MFG),
    ("we make indian sweets", "10734", "Manufacture of sweetmeats", MFG),
    ("mithai banane ka kaam", "10734", "Manufacture of sweetmeats", MFG),
    ("halwai making sweets", "10734", "Manufacture of sweetmeats", MFG),

    ("noodles and pasta manufacturing", "10740", "Manufacture of macaroni and noodles", MFG),
    ("we make noodles", "10740", "Manufacture of macaroni and noodles", MFG),
    ("vermicelli making unit", "10740", "Manufacture of macaroni and noodles", MFG),
    ("seviyan manufacturing", "10740", "Manufacture of macaroni and noodles", MFG),

    ("i make papad", "10796", "Manufacture of papads and appalam", MFG),
    ("papad making unit", "10796", "Manufacture of papads and appalam", MFG),
    ("papad and appalam production", "10796", "Manufacture of papads and appalam", MFG),

    ("tea processing", "10791", "Processing and blending of tea", MFG),
    ("tea packing and blending business", "10791", "Processing and blending of tea", MFG),
    ("chai packing unit", "10791", "Processing and blending of tea", MFG),

    ("spice grinding and processing", "10795", "Grinding and processing of spices", MFG),
    ("masala powder making", "10795", "Grinding and processing of spices", MFG),
    ("we grind and pack spices", "10795", "Grinding and processing of spices", MFG),
    ("masala manufacturing unit", "10795", "Grinding and processing of spices", MFG),
    ("mixed masala production", "10795", "Grinding and processing of spices", MFG),

    ("dairy product making ghee butter", "10504", "Manufacture of cream butter cheese ghee", MFG),
    ("i make ghee", "10504", "Manufacture of cream butter cheese ghee", MFG),
    ("ghee manufacturing unit", "10504", "Manufacture of cream butter cheese ghee", MFG),
    ("butter and paneer making", "10504", "Manufacture of cream butter cheese ghee", MFG),

    ("milk powder making", "10502", "Manufacture of milk powder", MFG),
    ("we make skimmed milk powder", "10502", "Manufacture of milk powder", MFG),

    ("ice cream manufacturing", "10505", "Manufacture of ice cream", MFG),
    ("we make ice cream kulfi", "10505", "Manufacture of ice cream", MFG),
    ("ice cream production unit", "10505", "Manufacture of ice cream", MFG),

    ("fish processing", "10204", "Processing and preserving of fish", MFG),
    ("we dry and pack fish", "10201", "Sun-drying of fish", MFG),
    ("dried fish business", "10201", "Sun-drying of fish", MFG),

    ("pickle making", "10306", "Manufacture of pickles and chutney", MFG),
    ("i make pickles achar", "10306", "Manufacture of pickles and chutney", MFG),
    ("achar making unit", "10306", "Manufacture of pickles and chutney", MFG),
    ("homemade pickle business", "10306", "Manufacture of pickles and chutney", MFG),
    ("chutney and pickle manufacturing", "10306", "Manufacture of pickles and chutney", MFG),

    ("fruit juice manufacturing", "10304", "Manufacture of fruit juices", MFG),
    ("we make fruit juice", "10304", "Manufacture of fruit juices", MFG),
    ("juice manufacturing unit", "10304", "Manufacture of fruit juices", MFG),

    ("jam jelly making", "10305", "Manufacture of sauces jams and jellies", MFG),
    ("we make jam jelly and sauce", "10305", "Manufacture of sauces jams and jellies", MFG),

    ("cattle feed manufacturing", "10801", "Manufacture of cattle feed", MFG),
    ("animal feed production", "10801", "Manufacture of cattle feed", MFG),

    # ========================
    # BEVERAGES
    # ========================
    ("aerated drinks manufacturing", "11041", "Manufacture of aerated drinks", MFG),
    ("soft drink manufacturing", "11045", "Manufacture of soft drinks", MFG),
    ("cold drink production", "11045", "Manufacture of soft drinks", MFG),
    ("mineral water plant", "11043", "Manufacture of mineral water", MFG),
    ("packaged drinking water business", "11043", "Manufacture of mineral water", MFG),
    ("we supply packaged water", "11043", "Manufacture of mineral water", MFG),

    # ========================
    # TEXTILES
    # ========================
    ("cotton yarn spinning", "13111", "Preparation and spinning of cotton fiber", MFG),
    ("spinning mill cotton", "13111", "Preparation and spinning of cotton fiber", MFG),
    ("cotton yarn manufacturing", "13111", "Preparation and spinning of cotton fiber", MFG),

    ("silk weaving", "13122", "Weaving of silk fabrics", MFG),
    ("saree weaving silk", "13122", "Weaving of silk fabrics", MFG),
    ("we weave silk fabric", "13122", "Weaving of silk fabrics", MFG),

    ("handloom weaving", "13121", "Weaving of cotton fabrics", MFG),
    ("we weave cotton fabric", "13121", "Weaving of cotton fabrics", MFG),
    ("power loom cotton fabric", "13121", "Weaving of cotton fabrics", MFG),
    ("cotton fabric weaving unit", "13121", "Weaving of cotton fabrics", MFG),

    ("garment stitching", "14101", "Manufacture of textile garments", MFG),
    ("readymade garments manufacturing", "14101", "Manufacture of textile garments", MFG),
    ("i make clothes and garments", "14101", "Manufacture of textile garments", MFG),
    ("garment factory", "14101", "Manufacture of textile garments", MFG),
    ("clothing manufacturing unit", "14101", "Manufacture of textile garments", MFG),

    ("tailoring unit", "14105", "Custom tailoring", MFG),
    ("i am a tailor", "14105", "Custom tailoring", MFG),
    ("custom tailoring shop", "14105", "Custom tailoring", MFG),
    ("darji ka kaam", "14105", "Custom tailoring", MFG),
    ("we do stitching and tailoring", "14105", "Custom tailoring", MFG),
    ("ladies tailor", "14105", "Custom tailoring", MFG),
    ("gents tailor shop", "14105", "Custom tailoring", MFG),

    ("embroidery work", "13991", "Embroidery work and making of laces", MFG),
    ("we do hand embroidery", "13991", "Embroidery work and making of laces", MFG),
    ("zardozi embroidery", "13991", "Embroidery work and making of laces", MFG),
    ("lace and embroidery business", "13991", "Embroidery work and making of laces", MFG),

    ("zari work business", "13992", "Zari work and ornamental trimmings", MFG),
    ("we do zari and gota work", "13992", "Zari work and ornamental trimmings", MFG),

    ("jute bag making", "13942", "Manufacture of cordage or rope of jute", MFG),
    ("jute products manufacturing", "13942", "Manufacture of cordage or rope of jute", MFG),

    ("carpet making", "13931", "Manufacture of carpets of cotton", MFG),
    ("we make handmade carpets", "13931", "Manufacture of carpets of cotton", MFG),
    ("carpet weaving unit", "13931", "Manufacture of carpets of cotton", MFG),

    ("knitting hosiery unit", "13913", "Manufacture of knitted synthetic fabrics", MFG),
    ("hosiery manufacturing", "13913", "Manufacture of knitted synthetic fabrics", MFG),

    # ========================
    # LEATHER & FOOTWEAR
    # ========================
    ("leather bag making", "15121", "Manufacture of travel goods and bags", MFG),
    ("i make leather bags and purses", "15121", "Manufacture of travel goods and bags", MFG),
    ("leather goods manufacturing", "15121", "Manufacture of travel goods and bags", MFG),
    ("handbag manufacturing", "15121", "Manufacture of travel goods and bags", MFG),

    ("shoe making", "15201", "Manufacture of leather footwear", MFG),
    ("i make shoes chappal", "15201", "Manufacture of leather footwear", MFG),
    ("footwear manufacturing", "15201", "Manufacture of leather footwear", MFG),
    ("joota banane ka kaam", "15201", "Manufacture of leather footwear", MFG),
    ("leather footwear production", "15201", "Manufacture of leather footwear", MFG),
    ("chappals and sandals making", "15201", "Manufacture of leather footwear", MFG),

    # ========================
    # WOOD & FURNITURE
    # ========================
    ("furniture making wood", "31001", "Manufacture of furniture of wood", MFG),
    ("wooden furniture unit", "31001", "Manufacture of furniture of wood", MFG),
    ("i make wooden furniture", "31001", "Manufacture of furniture of wood", MFG),
    ("furniture manufacturing", "31001", "Manufacture of furniture of wood", MFG),
    ("carpenter furniture shop", "31001", "Manufacture of furniture of wood", MFG),
    ("we manufacture wood furniture", "31001", "Manufacture of furniture of wood", MFG),

    ("plywood manufacturing", "16211", "Manufacture of plywood and veneer", MFG),
    ("plywood board making", "16211", "Manufacture of plywood and veneer", MFG),

    ("cardboard box making", "17023", "Manufacture of card board boxes", MFG),
    ("corrugated box manufacturing", "17023", "Manufacture of card board boxes", MFG),
    ("paper box and packaging", "17023", "Manufacture of card board boxes", MFG),
    ("we make cardboard boxes", "17023", "Manufacture of card board boxes", MFG),

    ("paper bag making", "17024", "Manufacture of sacks and bags of paper", MFG),
    ("we make paper carry bags", "17024", "Manufacture of sacks and bags of paper", MFG),

    # ========================
    # CHEMICALS & SOAP
    # ========================
    ("soap making", "20231", "Manufacture of soap all forms", MFG),
    ("i make soap at home", "20231", "Manufacture of soap all forms", MFG),
    ("handmade soap business", "20231", "Manufacture of soap all forms", MFG),
    ("soap manufacturing unit", "20231", "Manufacture of soap all forms", MFG),
    ("sabun banane ka kaam", "20231", "Manufacture of soap all forms", MFG),
    ("we manufacture soaps", "20231", "Manufacture of soap all forms", MFG),
    ("bathing soap production", "20231", "Manufacture of soap all forms", MFG),
    ("natural organic soap making", "20231", "Manufacture of soap all forms", MFG),
    ("liquid soap manufacturing", "20231", "Manufacture of soap all forms", MFG),

    ("detergent manufacturing", "20233", "Manufacture of detergent", MFG),
    ("we make washing powder", "20233", "Manufacture of detergent", MFG),
    ("detergent powder production", "20233", "Manufacture of detergent", MFG),
    ("washing powder manufacturing", "20233", "Manufacture of detergent", MFG),
    ("liquid detergent making", "20233", "Manufacture of detergent", MFG),

    ("paint manufacturing", "20221", "Manufacture of paints and varnishes", MFG),
    ("we make paints and varnish", "20221", "Manufacture of paints and varnishes", MFG),
    ("paint production unit", "20221", "Manufacture of paints and varnishes", MFG),

    ("fertilizer manufacturing", "20121", "Manufacture of urea and organic fertilizers", MFG),
    ("organic fertilizer making", "20121", "Manufacture of urea and organic fertilizers", MFG),

    ("agarbatti incense sticks making", "20238", "Manufacture of agarbatti", MFG),
    ("i make agarbatti", "20238", "Manufacture of agarbatti", MFG),
    ("incense stick manufacturing", "20238", "Manufacture of agarbatti", MFG),
    ("agarbatti production unit", "20238", "Manufacture of agarbatti", MFG),
    ("agarbatti banane ka kaam", "20238", "Manufacture of agarbatti", MFG),

    ("candle making", "20238", "Manufacture of agarbatti and similar products", MFG),
    ("mombatti banane ka kaam", "20238", "Manufacture of agarbatti and similar products", MFG),
    ("we make candles and wax products", "20238", "Manufacture of agarbatti and similar products", MFG),

    ("phenyl disinfectant making", "20212", "Manufacture of disinfectants", MFG),
    ("floor cleaner and phenyl production", "20212", "Manufacture of disinfectants", MFG),
    ("we make floor cleaning liquid", "20212", "Manufacture of disinfectants", MFG),

    ("hair oil shampoo manufacturing", "20236", "Manufacture of hair oil and shampoo", MFG),
    ("we make shampoo and hair products", "20236", "Manufacture of hair oil and shampoo", MFG),
    ("hair oil production unit", "20236", "Manufacture of hair oil and shampoo", MFG),

    ("cosmetics making", "20237", "Manufacture of cosmetics and toileteries", MFG),
    ("we manufacture cosmetics", "20237", "Manufacture of cosmetics and toileteries", MFG),
    ("beauty products manufacturing", "20237", "Manufacture of cosmetics and toileteries", MFG),
    ("cream lotion making", "20237", "Manufacture of cosmetics and toileteries", MFG),

    ("perfume making", "20234", "Manufacture of perfumes", MFG),
    ("attar and perfume business", "20234", "Manufacture of perfumes", MFG),
    ("we make perfumes and fragrances", "20234", "Manufacture of perfumes", MFG),

    ("toothpaste manufacturing", "20235", "Manufacture of dental hygiene preparations", MFG),
    ("we make toothpaste and toothpowder", "20235", "Manufacture of dental hygiene preparations", MFG),

    ("matchbox making", "20291", "Manufacture of matches", MFG),
    ("matchstick manufacturing", "20291", "Manufacture of matches", MFG),

    # ========================
    # PHARMA
    # ========================
    ("medicine manufacturing allopathic", "21002", "Manufacture of allopathic pharmaceutical preparations", MFG),
    ("tablet and capsule manufacturing", "21002", "Manufacture of allopathic pharmaceutical preparations", MFG),
    ("we make allopathic medicines", "21002", "Manufacture of allopathic pharmaceutical preparations", MFG),

    ("ayurvedic medicine making", "21003", "Manufacture of ayurvedic pharmaceutical preparations", MFG),
    ("herbal medicine production", "21003", "Manufacture of ayurvedic pharmaceutical preparations", MFG),
    ("ayurvedic product manufacturing", "21003", "Manufacture of ayurvedic pharmaceutical preparations", MFG),
    ("we make ayurvedic products", "21003", "Manufacture of ayurvedic pharmaceutical preparations", MFG),

    ("homeopathic medicine", "21004", "Manufacture of homoeopathic pharmaceutical preparations", MFG),
    ("we make homeopathy medicines", "21004", "Manufacture of homoeopathic pharmaceutical preparations", MFG),

    ("surgical cotton bandage making", "21006", "Manufacture of medical wadding gauze bandages", MFG),
    ("medical bandage manufacturing", "21006", "Manufacture of medical wadding gauze bandages", MFG),

    # ========================
    # PLASTICS & RUBBER
    # ========================
    ("plastic product manufacturing", "22201", "Manufacture of semi-finished plastic products", MFG),
    ("we make plastic sheets and boards", "22201", "Manufacture of semi-finished plastic products", MFG),

    ("plastic bag making", "22203", "Manufacture of plastic articles for packing", MFG),
    ("we make plastic bags and packets", "22203", "Manufacture of plastic articles for packing", MFG),
    ("plastic carry bag manufacturing", "22203", "Manufacture of plastic articles for packing", MFG),
    ("polythene bag making", "22203", "Manufacture of plastic articles for packing", MFG),

    ("plastic chair table making", "22202", "Manufacture of plastic household articles", MFG),
    ("plastic household items manufacturing", "22202", "Manufacture of plastic household articles", MFG),

    ("rubber product manufacturing", "22191", "Manufacture of rubber plates and tubes", MFG),
    ("tyre manufacturing", "22111", "Manufacture of rubber tyres for motor vehicles", MFG),
    ("rubber tyre production", "22111", "Manufacture of rubber tyres for motor vehicles", MFG),

    # ========================
    # CERAMICS & GLASS
    # ========================
    ("glass bangle making", "23106", "Manufacture of glass bangles", MFG),
    ("we make glass bangles", "23106", "Manufacture of glass bangles", MFG),
    ("churi banane ka kaam", "23106", "Manufacture of glass bangles", MFG),

    ("pottery making", "23931", "Manufacture of articles of porcelain and pottery", MFG),
    ("clay pottery business", "23931", "Manufacture of articles of porcelain and pottery", MFG),
    ("mitti ke bartan banane ka kaam", "23931", "Manufacture of articles of porcelain and pottery", MFG),
    ("ceramic pottery unit", "23931", "Manufacture of articles of porcelain and pottery", MFG),

    ("brick manufacturing", "23921", "Manufacture of bricks", MFG),
    ("we make bricks", "23921", "Manufacture of bricks", MFG),
    ("eent banane ka kaam", "23921", "Manufacture of bricks", MFG),
    ("fly ash brick making", "23921", "Manufacture of bricks", MFG),

    ("cement manufacturing", "23941", "Manufacture of clinkers and cement", MFG),

    # ========================
    # METALS
    # ========================
    ("steel manufacturing", "24103", "Manufacture of steel", MFG),
    ("iron and steel plant", "24101", "Manufacture of pig iron", MFG),
    ("aluminium product making", "24202", "Manufacture of aluminium products", MFG),
    ("copper wire making", "24201", "Manufacture of copper products", MFG),

    ("utensil making brass copper", "25994", "Manufacture of metal household articles", MFG),
    ("brassware manufacturing", "25994", "Manufacture of metal household articles", MFG),
    ("metal utensil production", "25994", "Manufacture of metal household articles", MFG),
    ("bartan banane ka kaam", "25994", "Manufacture of metal household articles", MFG),

    ("lock and key making", "25934", "Manufacture of padlocks and locks", MFG),
    ("we manufacture locks", "25934", "Manufacture of padlocks and locks", MFG),

    ("wire fencing making", "25993", "Manufacture of metal cable and wire", MFG),
    ("bolt nut manufacturing", "25991", "Manufacture of metal fasteners", MFG),
    ("nuts bolts and fasteners making", "25991", "Manufacture of metal fasteners", MFG),

    ("knife scissors making", "25931", "Manufacture of cutlery", MFG),
    ("cutlery manufacturing", "25931", "Manufacture of cutlery", MFG),
    ("we make knives and blades", "25931", "Manufacture of cutlery", MFG),

    ("tin box can making", "25992", "Manufacture of containers tins and cans", MFG),
    ("metal container manufacturing", "25992", "Manufacture of containers tins and cans", MFG),

    # ========================
    # ELECTRONICS
    # ========================
    ("mobile phone manufacturing", "26305", "Manufacture of pagers cellular phones", MFG),
    ("we assemble mobile phones", "26305", "Manufacture of pagers cellular phones", MFG),
    ("smartphone manufacturing unit", "26305", "Manufacture of pagers cellular phones", MFG),

    ("led bulb making", "27400", "Manufacture of electric lighting equipment", MFG),
    ("we make led lights and bulbs", "27400", "Manufacture of electric lighting equipment", MFG),
    ("led lamp manufacturing", "27400", "Manufacture of electric lighting equipment", MFG),

    ("electric fan manufacturing", "27503", "Manufacture of electric fans", MFG),
    ("fan manufacturing unit", "27503", "Manufacture of electric fans", MFG),

    ("wire and cable manufacturing", "27320", "Manufacture of wires and cables", MFG),
    ("electric cable production", "27320", "Manufacture of wires and cables", MFG),

    ("switch socket manufacturing", "27331", "Manufacture of switch and socket", MFG),
    ("electrical switch making", "27331", "Manufacture of switch and socket", MFG),

    ("battery manufacturing", "27201", "Manufacture of primary cells and batteries", MFG),
    ("we make batteries", "27201", "Manufacture of primary cells and batteries", MFG),

    ("computer assembly", "26201", "Manufacture of desktop and laptop computers", MFG),
    ("electronic component making", "26101", "Manufacture of electronic components", MFG),

    # ========================
    # AUTO
    # ========================
    ("auto parts manufacturing", "29301", "Manufacture of motor vehicle parts", MFG),
    ("car spare parts making", "29301", "Manufacture of motor vehicle parts", MFG),
    ("two wheeler manufacturing", "30911", "Manufacture of motorcycles and scooters", MFG),
    ("tractor manufacturing", "28211", "Manufacture of tractors for agriculture", MFG),
    ("bicycle manufacturing", "30921", "Manufacture of bicycles", MFG),

    # ========================
    # PRINTING
    # ========================
    ("printing press", "18112", "Printing of books and magazines", MFG),
    ("we run a printing press", "18112", "Printing of books and magazines", MFG),
    ("book printing business", "18112", "Printing of books and magazines", MFG),
    ("offset printing unit", "18112", "Printing of books and magazines", MFG),

    ("packaging printing", "18119", "Printing activities", MFG),
    ("label printing business", "18119", "Printing activities", MFG),

    # ========================
    # JEWELLERY
    # ========================
    ("gold jewellery making", "32111", "Manufacture of jewellery of gold silver", MFG),
    ("i make gold jewellery", "32111", "Manufacture of jewellery of gold silver", MFG),
    ("sone chandi ka gehna banane ka kaam", "32111", "Manufacture of jewellery of gold silver", MFG),
    ("silver jewellery manufacturing", "32111", "Manufacture of jewellery of gold silver", MFG),
    ("jewellery manufacturing unit", "32111", "Manufacture of jewellery of gold silver", MFG),

    ("imitation artificial jewellery", "32120", "Manufacture of imitation jewellery", MFG),
    ("we make fashion jewellery", "32120", "Manufacture of imitation jewellery", MFG),
    ("artificial jewellery manufacturing", "32120", "Manufacture of imitation jewellery", MFG),

    ("diamond cutting and polishing", "32112", "Working of diamonds and precious stones", MFG),
    ("diamond polishing unit", "32112", "Working of diamonds and precious stones", MFG),
    ("heera ghasai ka kaam", "32112", "Working of diamonds and precious stones", MFG),

    # ========================
    # TOYS & SPORTS
    # ========================
    ("toy making", "32401", "Manufacture of dolls and toy animals", MFG),
    ("wooden toy making", "32401", "Manufacture of dolls and toy animals", MFG),
    ("we make children toys", "32401", "Manufacture of dolls and toy animals", MFG),
    ("khilona banane ka kaam", "32401", "Manufacture of dolls and toy animals", MFG),

    ("sports goods manufacturing", "32300", "Manufacture of sports goods", MFG),
    ("we make cricket bats and sports equipment", "32300", "Manufacture of sports goods", MFG),

    # ========================
    # MEDICAL DEVICES
    # ========================
    ("surgical instrument making", "32504", "Manufacture of bone plates and syringes", MFG),
    ("medical device manufacturing", "32504", "Manufacture of medical instruments", MFG),

    # ========================
    # CONSTRUCTION
    # ========================
    ("building construction", "41001", "Construction of buildings", CON),
    ("house construction contractor", "41001", "Construction of buildings", CON),
    ("civil construction work", "41001", "Construction of buildings", CON),
    ("we build houses and flats", "41001", "Construction of buildings", CON),
    ("construction contractor", "41001", "Construction of buildings", CON),
    ("makan banane ka kaam", "41001", "Construction of buildings", CON),

    ("road construction", "42101", "Construction of highways and roads", CON),
    ("highway building contractor", "42101", "Construction of highways and roads", CON),
    ("road and bridge construction", "42101", "Construction of highways and roads", CON),

    ("electrical installation work", "43211", "Installation of electrical wiring", CON),
    ("we do electrical wiring", "43211", "Installation of electrical wiring", CON),
    ("electrician contractor", "43211", "Installation of electrical wiring", CON),

    ("plumbing work contractor", "43221", "Installation of plumbing", CON),
    ("plumber", "43221", "Installation of plumbing", CON),
    ("sanitary and plumbing work", "43221", "Installation of plumbing", CON),

    ("painting contractor", "43303", "Interior and exterior painting", CON),
    ("interior painting work", "43303", "Interior and exterior painting", CON),
    ("house painting business", "43303", "Interior and exterior painting", CON),

    ("interior design work", "43303", "Interior painting and decorating", CON),
    ("we do interior decoration", "43303", "Interior painting and decorating", CON),

    # ========================
    # TRADE
    # ========================
    ("grocery shop", "47211", "Retail sale of cereals and pulses", TRD),
    ("kirana store", "47211", "Retail sale of food products", TRD),
    ("general store", "47190", "Other retail sale in non-specialized stores", TRD),
    ("we sell groceries", "47211", "Retail sale of cereals and pulses", TRD),
    ("kiryana shop", "47211", "Retail sale of cereals and pulses", TRD),
    ("small grocery shop", "47211", "Retail sale of cereals and pulses", TRD),
    ("provision store", "47190", "Other retail sale in non-specialized stores", TRD),

    ("mobile phone shop", "47414", "Retail sale of telecommunication equipment", TRD),
    ("we sell mobile phones", "47414", "Retail sale of telecommunication equipment", TRD),
    ("mobile recharge and phone shop", "47414", "Retail sale of telecommunication equipment", TRD),

    ("electronic goods shop", "47420", "Retail sale of audio and video equipment", TRD),
    ("we sell electronics", "47420", "Retail sale of audio and video equipment", TRD),
    ("electronics retail shop", "47420", "Retail sale of audio and video equipment", TRD),

    ("clothing shop", "47711", "Retail sale of readymade garments", TRD),
    ("readymade garment shop", "47711", "Retail sale of readymade garments", TRD),
    ("we sell clothes", "47711", "Retail sale of readymade garments", TRD),
    ("kapde ki dukan", "47711", "Retail sale of readymade garments", TRD),

    ("medicine pharmacy", "47721", "Retail sale of pharmaceuticals", TRD),
    ("we run a medical store", "47721", "Retail sale of pharmaceuticals", TRD),
    ("chemist shop", "47721", "Retail sale of pharmaceuticals", TRD),
    ("dawai ki dukan", "47721", "Retail sale of pharmaceuticals", TRD),

    ("jewellery shop", "47733", "Retail sale of jewellery", TRD),
    ("we sell gold and silver jewellery", "47733", "Retail sale of jewellery", TRD),
    ("sona chandi ki dukan", "47733", "Retail sale of jewellery", TRD),

    ("petrol pump", "47300", "Retail sale of automotive fuel", TRD),
    ("we run a petrol station", "47300", "Retail sale of automotive fuel", TRD),
    ("fuel station", "47300", "Retail sale of automotive fuel", TRD),

    ("book stationery shop", "47613", "Retail sale of stationery", TRD),
    ("we sell books and stationery", "47613", "Retail sale of stationery", TRD),
    ("kitab ki dukan", "47613", "Retail sale of stationery", TRD),

    ("hardware shop", "47522", "Retail sale of hardware", TRD),
    ("we sell hardware and tools", "47522", "Retail sale of hardware", TRD),
    ("hardware store", "47522", "Retail sale of hardware", TRD),

    ("vegetable fruit shop", "47212", "Retail sale of fresh fruit and vegetables", TRD),
    ("sabzi mandi stall", "47212", "Retail sale of fresh fruit and vegetables", TRD),
    ("we sell vegetables and fruits", "47212", "Retail sale of fresh fruit and vegetables", TRD),

    ("furniture shop", "47591", "Retail sale of household furniture", TRD),
    ("we sell furniture", "47591", "Retail sale of household furniture", TRD),

    ("online selling ecommerce", "47912", "Retail sale via e-commerce", TRD),
    ("we sell online on amazon flipkart", "47912", "Retail sale via e-commerce", TRD),
    ("ecommerce business", "47912", "Retail sale via e-commerce", TRD),
    ("online retail business", "47912", "Retail sale via e-commerce", TRD),

    ("vehicle repair garage", "45200", "Maintenance and repair of motor vehicles", TRD),
    ("we repair cars and vehicles", "45200", "Maintenance and repair of motor vehicles", TRD),
    ("auto garage mechanic", "45200", "Maintenance and repair of motor vehicles", TRD),
    ("car servicing workshop", "45200", "Maintenance and repair of motor vehicles", TRD),

    ("bike repair shop", "45403", "Maintenance of motorcycles and scooters", TRD),
    ("two wheeler repair workshop", "45403", "Maintenance of motorcycles and scooters", TRD),
    ("motorcycle mechanic", "45403", "Maintenance of motorcycles and scooters", TRD),

    # ========================
    # TRANSPORT
    # ========================
    ("truck transport business", "49231", "Motorised road freight transport", TRN),
    ("we do truck transport", "49231", "Motorised road freight transport", TRN),
    ("road freight transport", "49231", "Motorised road freight transport", TRN),
    ("goods transport by truck", "49231", "Motorised road freight transport", TRN),
    ("transport contractor", "49231", "Motorised road freight transport", TRN),

    ("taxi cab service", "49224", "Taxi operation", TRN),
    ("we run taxis", "49224", "Taxi operation", TRN),
    ("auto rickshaw", "49224", "Taxi operation", TRN),
    ("cab service business", "49224", "Taxi operation", TRN),
    ("we operate auto rickshaw", "49224", "Taxi operation", TRN),

    ("bus transport service", "49221", "Long distance bus services", TRN),
    ("we run bus service", "49221", "Long distance bus services", TRN),

    ("courier service", "53200", "Courier activities", TRN),
    ("parcel delivery business", "53200", "Courier activities", TRN),
    ("we deliver parcels", "53200", "Courier activities", TRN),
    ("courier and logistics", "53200", "Courier activities", TRN),

    ("cold storage warehouse", "52101", "Warehousing of refrigerated goods", TRN),
    ("cold chain storage business", "52101", "Warehousing of refrigerated goods", TRN),
    ("warehouse storage facility", "52102", "Warehousing non-refrigerated", TRN),
    ("we provide warehousing", "52102", "Warehousing non-refrigerated", TRN),

    ("travel agency", "52291", "Activities of travel agents", TRN),
    ("we are a travel agent", "52291", "Activities of travel agents", TRN),
    ("tour and travel agency", "52291", "Activities of travel agents", TRN),

    # ========================
    # HOSPITALITY
    # ========================
    ("hotel restaurant", "55101", "Hotels and motels providing lodging", HOS),
    ("we run a hotel", "55101", "Hotels and motels providing lodging", HOS),
    ("lodge and hotel business", "55101", "Hotels and motels providing lodging", HOS),

    ("dhaba food stall", "56101", "Restaurants", HOS),
    ("we run a dhaba", "56101", "Restaurants", HOS),
    ("restaurant business", "56101", "Restaurants", HOS),
    ("khana khilane ka dhaba", "56101", "Restaurants", HOS),

    ("tiffin catering service", "56291", "Activities of food service contractors", HOS),
    ("home tiffin service", "56291", "Activities of food service contractors", HOS),
    ("we provide tiffin to offices", "56291", "Activities of food service contractors", HOS),
    ("meal delivery catering", "56291", "Activities of food service contractors", HOS),

    ("canteen mess", "56292", "Operation of canteens", HOS),
    ("factory canteen", "56292", "Operation of canteens", HOS),
    ("we run an office canteen", "56292", "Operation of canteens", HOS),

    ("juice shop", "56303", "Fruit juice bars", HOS),
    ("we sell fresh juice", "56303", "Fruit juice bars", HOS),
    ("fruit juice centre", "56303", "Fruit juice bars", HOS),

    ("tea stall", "56302", "Tea and coffee shops", HOS),
    ("chai ki tapri", "56302", "Tea and coffee shops", HOS),
    ("we run a tea shop", "56302", "Tea and coffee shops", HOS),

    ("fast food centre", "56102", "Cafeterias and fast food restaurants", HOS),
    ("we sell fast food", "56102", "Cafeterias and fast food restaurants", HOS),
    ("quick service food outlet", "56102", "Cafeterias and fast food restaurants", HOS),

    ("event catering", "56210", "Event catering", HOS),
    ("we do catering for events", "56210", "Event catering", HOS),
    ("wedding catering service", "56210", "Event catering", HOS),

    # ========================
    # IT SERVICES
    # ========================
    ("software development", "62011", "Writing and modifying computer programs", IT),
    ("we develop software", "62011", "Writing and modifying computer programs", IT),
    ("app development company", "62011", "Writing and modifying computer programs", IT),
    ("custom software development", "62011", "Writing and modifying computer programs", IT),

    ("web design", "62012", "Web page designing", IT),
    ("we make websites", "62012", "Web page designing", IT),
    ("website development", "62012", "Web page designing", IT),
    ("web development agency", "62012", "Web page designing", IT),

    ("mobile app development", "62011", "Writing computer programs", IT),
    ("we build mobile apps", "62011", "Writing computer programs", IT),

    ("it consulting", "62020", "Computer consultancy", IT),
    ("it services company", "62020", "Computer consultancy", IT),
    ("computer consultancy", "62020", "Computer consultancy", IT),

    ("data entry work", "63114", "Providing data entry services", IT),
    ("we do data entry", "63114", "Providing data entry services", IT),
    ("data entry operator", "63114", "Providing data entry services", IT),

    ("cyber cafe", "63992", "Activities of cyber cafe", IT),
    ("internet cafe", "63992", "Activities of cyber cafe", IT),
    ("we run a cyber cafe", "63992", "Activities of cyber cafe", IT),

    ("call centre bpo", "82200", "Activities of call centres", IT),
    ("we run a call center", "82200", "Activities of call centres", IT),
    ("bpo services", "82200", "Activities of call centres", IT),

    ("internet service provider", "61104", "Activities providing internet access", IT),
    ("cable tv operator", "61103", "Activities of cable operators", IT),
    ("we are a cable operator", "61103", "Activities of cable operators", IT),

    # ========================
    # FINANCE
    # ========================
    ("money lending", "64920", "Other credit granting", FIN),
    ("we give loans to people", "64920", "Other credit granting", FIN),
    ("moneylending business", "64920", "Other credit granting", FIN),

    ("chit fund", "64910", "Financial leasing", FIN),
    ("we run a chit fund", "64910", "Financial leasing", FIN),

    ("insurance agent", "66220", "Activities of insurance agents and brokers", FIN),
    ("we sell insurance policies", "66220", "Activities of insurance agents and brokers", FIN),
    ("insurance broker", "66220", "Activities of insurance agents and brokers", FIN),

    ("stock broker", "66120", "Security and commodity contracts brokerage", FIN),
    ("we do share trading", "66120", "Security and commodity contracts brokerage", FIN),
    ("share broker firm", "66120", "Security and commodity contracts brokerage", FIN),

    ("tax consultant", "69202", "Tax consultancy", FIN),
    ("we do tax filing and GST", "69202", "Tax consultancy", FIN),
    ("income tax consultant", "69202", "Tax consultancy", FIN),
    ("gst and tax services", "69202", "Tax consultancy", FIN),

    ("accountant ca firm", "69201", "Accounting and auditing activities", FIN),
    ("chartered accountant firm", "69201", "Accounting and auditing activities", FIN),
    ("we do accounting and audit", "69201", "Accounting and auditing activities", FIN),

    # ========================
    # REAL ESTATE
    # ========================
    ("property dealer", "68200", "Real estate activities on fee basis", RE),
    ("real estate agent broker", "68200", "Real estate activities on fee basis", RE),
    ("property buying and selling", "68200", "Real estate activities on fee basis", RE),
    ("we are property dealers", "68200", "Real estate activities on fee basis", RE),
    ("zameen jaidaad ka kaam", "68200", "Real estate activities on fee basis", RE),

    # ========================
    # PROFESSIONAL SERVICES
    # ========================
    ("advocate lawyer", "69100", "Legal activities", PRO),
    ("we are a law firm", "69100", "Legal activities", PRO),
    ("legal services", "69100", "Legal activities", PRO),
    ("vakil ka kaam", "69100", "Legal activities", PRO),

    ("architect firm", "71100", "Architectural and engineering activities", PRO),
    ("civil engineer consultant", "71100", "Architectural and engineering activities", PRO),
    ("we provide architectural services", "71100", "Architectural and engineering activities", PRO),

    ("advertising agency", "73100", "Advertising", PRO),
    ("we do advertising and marketing", "73100", "Advertising", PRO),
    ("digital marketing agency", "73100", "Advertising", PRO),

    ("photography studio", "74201", "Commercial photograph production", PRO),
    ("we do photography", "74201", "Commercial photograph production", PRO),
    ("wedding photography", "74201", "Commercial photograph production", PRO),
    ("foto studio", "74201", "Commercial photograph production", PRO),

    ("graphic design", "74103", "Services of graphic designers", PRO),
    ("we do graphic design work", "74103", "Services of graphic designers", PRO),
    ("logo and design services", "74103", "Services of graphic designers", PRO),

    ("fashion designer", "74101", "Fashion design", PRO),
    ("we design clothes", "74101", "Fashion design", PRO),

    ("veterinary doctor", "75000", "Veterinary activities", PRO),
    ("animal doctor clinic", "75000", "Veterinary activities", PRO),
    ("pashu chikitsa", "75000", "Veterinary activities", PRO),

    # ========================
    # EDUCATION
    # ========================
    ("coaching class tuition", "85491", "Academic tutoring services", EDU),
    ("we run a coaching class", "85491", "Academic tutoring services", EDU),
    ("tuition centre", "85491", "Academic tutoring services", EDU),
    ("coaching institute", "85491", "Academic tutoring services", EDU),
    ("private tuitions", "85491", "Academic tutoring services", EDU),

    ("private school", "85211", "General school education first stage", EDU),
    ("we run a school", "85211", "General school education first stage", EDU),
    ("vidyalaya school", "85211", "General school education first stage", EDU),

    ("college education", "85301", "Higher education", EDU),

    ("skill training institute iti", "85221", "Technical and vocational education", EDU),
    ("vocational training centre", "85221", "Technical and vocational education", EDU),
    ("iti training institute", "85221", "Technical and vocational education", EDU),

    ("driving school", "85223", "Professional motor driving school", EDU),
    ("we run a driving school", "85223", "Professional motor driving school", EDU),
    ("motor driving institute", "85223", "Professional motor driving school", EDU),

    ("dance music class", "85420", "Cultural education", EDU),
    ("we teach dance and music", "85420", "Cultural education", EDU),
    ("music and arts academy", "85420", "Cultural education", EDU),

    # ========================
    # HEALTHCARE
    # ========================
    ("hospital clinic", "86100", "Hospital activities", HC),
    ("we run a hospital", "86100", "Hospital activities", HC),
    ("nursing home", "86100", "Hospital activities", HC),
    ("private hospital", "86100", "Hospital activities", HC),

    ("doctor medical practice", "86201", "Medical practice activities", HC),
    ("we are a medical clinic", "86201", "Medical practice activities", HC),
    ("doctor clinic", "86201", "Medical practice activities", HC),
    ("general practitioner clinic", "86201", "Medical practice activities", HC),

    ("dentist", "86202", "Dental practice activities", HC),
    ("dental clinic", "86202", "Dental practice activities", HC),
    ("we run a dental clinic", "86202", "Dental practice activities", HC),

    ("ayurvedic clinic", "86901", "Activities of Ayurveda practitioners", HC),
    ("we run an ayurvedic centre", "86901", "Activities of Ayurveda practitioners", HC),

    ("homeopathy clinic", "86903", "Activities of homeopaths", HC),
    ("homeopathic doctor clinic", "86903", "Activities of homeopaths", HC),

    ("diagnostic lab pathology", "86905", "Activities of diagnostic laboratories", HC),
    ("we run a pathology lab", "86905", "Activities of diagnostic laboratories", HC),
    ("blood test lab", "86905", "Activities of diagnostic laboratories", HC),

    # ========================
    # PERSONAL SERVICES
    # ========================
    ("beauty parlour salon", "96020", "Hairdressing and other beauty treatment", SVC),
    ("we run a beauty salon", "96020", "Hairdressing and other beauty treatment", SVC),
    ("hair cutting salon", "96020", "Hairdressing and other beauty treatment", SVC),
    ("ladies beauty parlour", "96020", "Hairdressing and other beauty treatment", SVC),
    ("parlour and spa", "96020", "Hairdressing and other beauty treatment", SVC),
    ("salon business", "96020", "Hairdressing and other beauty treatment", SVC),
    ("nai ki dukan", "96020", "Hairdressing and other beauty treatment", SVC),

    ("dry cleaning laundry", "96010", "Washing and dry cleaning of textiles", SVC),
    ("we run a laundry", "96010", "Washing and dry cleaning of textiles", SVC),
    ("dry cleaning service", "96010", "Washing and dry cleaning of textiles", SVC),
    ("dhobighat laundry", "96010", "Washing and dry cleaning of textiles", SVC),

    ("repair shop electronics", "95210", "Repair of consumer electronics", SVC),
    ("tv and electronics repair", "95210", "Repair of consumer electronics", SVC),
    ("we repair electronic items", "95210", "Repair of consumer electronics", SVC),

    ("mobile phone repair", "95120", "Repair of communication equipment", SVC),
    ("we repair mobile phones", "95120", "Repair of communication equipment", SVC),
    ("phone repair shop", "95120", "Repair of communication equipment", SVC),
    ("mobile repairing centre", "95120", "Repair of communication equipment", SVC),

    ("bicycle repair", "95291", "Repair of bicycles", SVC),
    ("we repair cycles", "95291", "Repair of bicycles", SVC),
    ("cycle repair shop", "95291", "Repair of bicycles", SVC),

    ("cobbler shoe repair", "95230", "Repair of footwear and leather goods", SVC),
    ("we repair shoes", "95230", "Repair of footwear and leather goods", SVC),
    ("mochi ka kaam", "95230", "Repair of footwear and leather goods", SVC),
    ("footwear repair shop", "95230", "Repair of footwear and leather goods", SVC),

    ("pest control", "81299", "Other building and industrial cleaning", SVC),
    ("we do pest control", "81299", "Other building and industrial cleaning", SVC),
    ("pest control services", "81299", "Other building and industrial cleaning", SVC),

    ("security guard agency", "80100", "Private security activities", SVC),
    ("we provide security guards", "80100", "Private security activities", SVC),
    ("security services company", "80100", "Private security activities", SVC),

    ("cleaning services", "81210", "General cleaning of buildings", SVC),
    ("housekeeping and cleaning", "81210", "General cleaning of buildings", SVC),
    ("we provide cleaning staff", "81210", "General cleaning of buildings", SVC),

    ("event management", "82300", "Organization of conventions and trade shows", SVC),
    ("we manage events", "82300", "Organization of conventions and trade shows", SVC),
    ("wedding event management", "82300", "Organization of conventions and trade shows", SVC),

    ("placement agency hr", "78100", "Activities of employment placement agencies", SVC),
    ("we provide manpower", "78100", "Activities of employment placement agencies", SVC),
    ("recruitment agency", "78100", "Activities of employment placement agencies", SVC),

    ("photocopying shop", "82191", "Photocopying and duplicating services", SVC),
    ("xerox and printing shop", "82191", "Photocopying and duplicating services", SVC),

    ("machine repair workshop", "33121", "Repair of engines and turbines", MFG),
    ("we repair industrial machines", "33121", "Repair of engines and turbines", MFG),

    # ========================
    # ENERGY
    # ========================
    ("solar energy plant", "35105", "Electric power generation using solar energy", ENE),
    ("we install solar panels", "35105", "Electric power generation using solar energy", ENE),
    ("solar power business", "35105", "Electric power generation using solar energy", ENE),
    ("solar rooftop installation", "35105", "Electric power generation using solar energy", ENE),

    ("wind energy farm", "35106", "Electric power generation non conventional", ENE),
    ("electricity distribution", "35109", "Collection and distribution of electric energy", ENE),
    ("gas agency cylinder", "35202", "Distribution and sale of gaseous fuels", ENE),
    ("lpg gas agency", "35202", "Distribution and sale of gaseous fuels", ENE),
    ("we run a gas distributor", "35202", "Distribution and sale of gaseous fuels", ENE),
]

# ============================================================
# DIVERSE AUGMENTATION — 14 prefix/suffix combos
# ============================================================
PREFIXES = [
    "", "", "", "", "",
    "we are into ",
    "our business is ",
    "i am doing ",
    "we deal in ",
    "my work involves ",
    "the company is engaged in ",
    "business activity: ",
    "i run a ",
    "we have a ",
    "i own a ",
    "i do ",
    "we do ",
    "our unit is involved in ",
    "we specialize in ",
    "i am working in ",
    "we specialize in ",
    "i am a ",
    "my small business is ",
    "we are manufacturers of ",
    "i sell ",
    "we provide ",

]

SUFFIXES = [
    "", "", "", "", "",
    " in india",
    " for local market",
    " as msme",
    " at home",
    " in my village",
    " on small scale",
    " for export",
    " for domestic use",
    " as a home business",
    " as micro enterprise",
    " unit",
    " business",
    " near me",
    " since many years",

]

def augment(text, nic, label, division, n=28):
    results = []
    seen = {text.strip().lower()}
    results.append((text.strip(), nic, label, division))

    attempts = 0
    while len(results) < n and attempts < n * 5:
        p = random.choice(PREFIXES)
        s = random.choice(SUFFIXES)
        new_text = (p + text + s).strip()
        if new_text.lower() not in seen:
            results.append((new_text, nic, label, division))
            seen.add(new_text.lower())
        attempts += 1

    return results

# ============================================================
# BUILD DATASET
# ============================================================
all_rows = []
for (text, nic, label, division) in RAW_DATA:
    all_rows.extend(augment(text, nic, label, division))

random.shuffle(all_rows)

# ============================================================
# SAVE
# ============================================================
csv_path = "data/industries.csv"
import os
os.makedirs("data", exist_ok=True)

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["text", "nic_code", "nic_label", "division"])
    writer.writeheader()
    for (text, nic, label, div) in all_rows:
        writer.writerow({"text": text, "nic_code": nic, "nic_label": label, "division": div})

nic_lookup = {}
for (_, nic, label, div) in RAW_DATA:
    nic_lookup[nic] = {"label": label, "division": div}

with open("data/nic_lookup.json", "w", encoding="utf-8") as f:
    json.dump(nic_lookup, f, indent=2, ensure_ascii=False)

# Stats
div_counts = Counter(d for _, _, _, d in all_rows)
nic_counts  = Counter(n for _, n, _, _ in all_rows)

print("=" * 60)
print(f"  DATASET v2 COMPLETE")
print(f"  Total samples     : {len(all_rows)}")
print(f"  Unique NIC codes  : {len(set(n for _,n,_,_ in all_rows))}")
print(f"  Avg per NIC code  : {len(all_rows) / len(set(n for _,n,_,_ in all_rows)):.1f}")
print()
print("  Samples per division:")
for div, cnt in sorted(div_counts.items(), key=lambda x: -x[1]):
    print(f"    {div:<25} {cnt}")
print("=" * 60)