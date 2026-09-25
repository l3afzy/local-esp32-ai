"""Generates the templated part of the facts (capitals, elements, countries, ...).

    pip install mendeleev babel phonenumbers scipy pycountry
    python data/more_facts.py > data/facts_generated.tsv

Hand-written facts live in facts.tsv; templated ones are generated here so
every entry gets the same set of phrasings. Bulk data comes from permissively
licensed packages rather than memory:

    mendeleev (MIT)        element properties
    babel / CLDR (BSD)     country currencies
    phonenumbers (Apache)  country calling codes
    scipy (BSD)            CODATA physical constants
    pycountry (LGPL)       US state and Canadian province abbreviations
"""

import unicodedata

COUNTRY_CAPITALS = {
    "afghanistan": "Kabul.", "albania": "Tirana.", "algeria": "Algiers.",
    "andorra": "Andorra la Vella.", "angola": "Luanda.", "antigua and barbuda": "Saint John's.",
    "armenia": "Yerevan.", "azerbaijan": "Baku.", "the bahamas": "Nassau.", "bahrain": "Manama.",
    "bangladesh": "Dhaka.", "barbados": "Bridgetown.", "belarus": "Minsk.", "belize": "Belmopan.",
    "benin": "Porto-Novo.", "bhutan": "Thimphu.",
    "bolivia": "Sucre; La Paz is the seat of government.",
    "bosnia and herzegovina": "Sarajevo.", "botswana": "Gaborone.",
    "brunei": "Bandar Seri Begawan.", "bulgaria": "Sofia.", "burkina faso": "Ouagadougou.",
    "burundi": "Gitega.", "cambodia": "Phnom Penh.", "cameroon": "Yaounde.",
    "cape verde": "Praia.", "the central african republic": "Bangui.", "chad": "N'Djamena.",
    "comoros": "Moroni.", "the republic of the congo": "Brazzaville.",
    "the democratic republic of the congo": "Kinshasa.", "costa rica": "San Jose.",
    "ivory coast": "Yamoussoukro.", "croatia": "Zagreb.", "cuba": "Havana.",
    "cyprus": "Nicosia.", "the czech republic": "Prague.", "djibouti": "Djibouti.",
    "dominica": "Roseau.", "the dominican republic": "Santo Domingo.", "ecuador": "Quito.",
    "el salvador": "San Salvador.", "equatorial guinea": "Malabo.", "eritrea": "Asmara.",
    "estonia": "Tallinn.", "eswatini": "Mbabane.", "ethiopia": "Addis Ababa.", "fiji": "Suva.",
    "gabon": "Libreville.", "the gambia": "Banjul.",
    "georgia": "Tbilisi. The US state's capital is Atlanta.",
    "ghana": "Accra.", "grenada": "Saint George's.", "guatemala": "Guatemala City.",
    "guinea": "Conakry.", "guinea-bissau": "Bissau.", "guyana": "Georgetown.",
    "haiti": "Port-au-Prince.", "honduras": "Tegucigalpa.", "hungary": "Budapest.",
    "iceland": "Reykjavik.", "iraq": "Baghdad.", "jamaica": "Kingston.", "jordan": "Amman.",
    "kazakhstan": "Astana.", "kiribati": "South Tarawa.", "kuwait": "Kuwait City.",
    "kyrgyzstan": "Bishkek.", "laos": "Vientiane.", "latvia": "Riga.", "lebanon": "Beirut.",
    "lesotho": "Maseru.", "liberia": "Monrovia.", "libya": "Tripoli.",
    "liechtenstein": "Vaduz.", "lithuania": "Vilnius.", "luxembourg": "Luxembourg.",
    "madagascar": "Antananarivo.", "malawi": "Lilongwe.", "malaysia": "Kuala Lumpur.",
    "the maldives": "Male.", "mali": "Bamako.", "malta": "Valletta.",
    "the marshall islands": "Majuro.", "mauritania": "Nouakchott.", "mauritius": "Port Louis.",
    "micronesia": "Palikir.", "moldova": "Chisinau.", "monaco": "Monaco.",
    "mongolia": "Ulaanbaatar.", "montenegro": "Podgorica.", "mozambique": "Maputo.",
    "myanmar": "Naypyidaw.", "namibia": "Windhoek.",
    "nauru": "No official capital; government sits in Yaren.",
    "nepal": "Kathmandu.", "nicaragua": "Managua.", "niger": "Niamey.",
    "north korea": "Pyongyang.", "north macedonia": "Skopje.", "oman": "Muscat.",
    "palau": "Ngerulmud.", "panama": "Panama City.", "papua new guinea": "Port Moresby.",
    "paraguay": "Asuncion.", "qatar": "Doha.", "romania": "Bucharest.", "rwanda": "Kigali.",
    "saint kitts and nevis": "Basseterre.", "saint lucia": "Castries.",
    "saint vincent and the grenadines": "Kingstown.", "samoa": "Apia.",
    "san marino": "San Marino.", "sao tome and principe": "Sao Tome.", "senegal": "Dakar.",
    "serbia": "Belgrade.", "seychelles": "Victoria.", "sierra leone": "Freetown.",
    "singapore": "Singapore.", "slovakia": "Bratislava.", "slovenia": "Ljubljana.",
    "the solomon islands": "Honiara.", "somalia": "Mogadishu.", "south sudan": "Juba.",
    "sri lanka": "Sri Jayawardenepura Kotte.", "sudan": "Khartoum.",
    "suriname": "Paramaribo.", "syria": "Damascus.", "taiwan": "Taipei.",
    "tajikistan": "Dushanbe.", "tanzania": "Dodoma.", "east timor": "Dili.", "togo": "Lome.",
    "tonga": "Nuku'alofa.", "trinidad and tobago": "Port of Spain.", "tunisia": "Tunis.",
    "turkmenistan": "Ashgabat.", "tuvalu": "Funafuti.", "uganda": "Kampala.",
    "the united arab emirates": "Abu Dhabi.", "uruguay": "Montevideo.",
    "uzbekistan": "Tashkent.", "vanuatu": "Port Vila.", "venezuela": "Caracas.",
    "yemen": "Sanaa.", "zambia": "Lusaka.", "zimbabwe": "Harare.",
    "vatican city": "Vatican City.", "kosovo": "Pristina.", "greenland": "Nuuk.",
    "scotland": "Edinburgh.", "wales": "Cardiff.", "northern ireland": "Belfast.",
}

EXTRA_NAMES = {  # other ways people name the same place
    "the czech republic": ["czechia"], "ivory coast": ["cote d'ivoire"],
    "east timor": ["timor-leste"], "the united arab emirates": ["uae"],
    "the democratic republic of the congo": ["drc", "dr congo"],
    "the republic of the congo": ["congo"], "eswatini": ["swaziland"],
    "myanmar": ["burma"], "cape verde": ["cabo verde"],
}

US_STATE_CAPITALS = {
    "alabama": "Montgomery.", "alaska": "Juneau.", "arizona": "Phoenix.",
    "arkansas": "Little Rock.", "california": "Sacramento.", "colorado": "Denver.",
    "connecticut": "Hartford.", "delaware": "Dover.", "florida": "Tallahassee.",
    "hawaii": "Honolulu.", "idaho": "Boise.", "illinois": "Springfield.",
    "indiana": "Indianapolis.", "iowa": "Des Moines.", "kansas": "Topeka.",
    "kentucky": "Frankfort.", "louisiana": "Baton Rouge.", "maine": "Augusta.",
    "maryland": "Annapolis.", "massachusetts": "Boston.", "michigan": "Lansing.",
    "minnesota": "Saint Paul.", "mississippi": "Jackson.", "missouri": "Jefferson City.",
    "montana": "Helena.", "nebraska": "Lincoln.", "nevada": "Carson City.",
    "new hampshire": "Concord.", "new jersey": "Trenton.", "new mexico": "Santa Fe.",
    "new york": "Albany.", "north carolina": "Raleigh.", "north dakota": "Bismarck.",
    "ohio": "Columbus.", "oklahoma": "Oklahoma City.", "oregon": "Salem.",
    "pennsylvania": "Harrisburg.", "rhode island": "Providence.",
    "south carolina": "Columbia.", "south dakota": "Pierre.", "tennessee": "Nashville.",
    "texas": "Austin.", "utah": "Salt Lake City.", "vermont": "Montpelier.",
    "virginia": "Richmond.", "washington": "Olympia.", "west virginia": "Charleston.",
    "wisconsin": "Madison.", "wyoming": "Cheyenne.",
}

# Discoverers, curated by hand: the mendeleev package's list has misspellings
# and several wrong entries, so only well-documented cases are included.
DISCOVERED = {  # element: (who, when)
    "hydrogen": ("Henry Cavendish.", "1766."),
    "helium": ("Janssen and Lockyer, in the Sun's spectrum.", "1868."),
    "nitrogen": ("Daniel Rutherford.", "1772."),
    "oxygen": ("Carl Scheele and Joseph Priestley.", "1774."),
    "fluorine": ("Henri Moissan isolated it.", "1886."),
    "sodium": ("Humphry Davy.", "1807."), "potassium": ("Humphry Davy.", "1807."),
    "calcium": ("Humphry Davy.", "1808."),
    "phosphorus": ("Hennig Brand.", "1669."), "chlorine": ("Carl Wilhelm Scheele.", "1774."),
    "argon": ("Lord Rayleigh and William Ramsay.", "1894."),
    "neon": ("William Ramsay and Morris Travers.", "1898."),
    "krypton": ("William Ramsay and Morris Travers.", "1898."),
    "xenon": ("William Ramsay and Morris Travers.", "1898."),
    "titanium": ("William Gregor.", "1791."), "cobalt": ("Georg Brandt.", "About 1735."),
    "nickel": ("Axel Fredrik Cronstedt.", "1751."),
    "gallium": ("Paul-Emile Lecoq de Boisbaudran.", "1875."),
    "germanium": ("Clemens Winkler.", "1886."),
    "rubidium": ("Robert Bunsen and Gustav Kirchhoff.", "1861."),
    "cesium": ("Robert Bunsen and Gustav Kirchhoff.", "1860."),
    "technetium": ("Carlo Perrier and Emilio Segre.", "1937."),
    "iodine": ("Bernard Courtois.", "1811."), "hafnium": ("Dirk Coster and George de Hevesy.", "1923."),
    "tungsten": ("The Elhuyar brothers isolated it.", "1783."),
    "thallium": ("William Crookes.", "1861."),
    "polonium": ("Pierre and Marie Curie.", "1898."), "radium": ("Pierre and Marie Curie.", "1898."),
    "francium": ("Marguerite Perey.", "1939."), "uranium": ("Martin Heinrich Klaproth.", "1789."),
    "neptunium": ("Edwin McMillan and Philip Abelson.", "1940."),
    "plutonium": ("Glenn Seaborg and colleagues.", "1940."),
}
ANCIENT = {"carbon", "sulfur", "iron", "copper", "silver", "tin", "gold", "mercury", "lead",
           "antimony"}
ELEMENT_ALIASES = {"aluminum": ["aluminium"], "cesium": ["caesium"], "sulfur": ["sulphur"]}
SERIES = {  # mendeleev series -> (type answer, "is it a metal" answer)
    "Nonmetals": ("A nonmetal.", "No, a nonmetal."),
    "Noble gases": ("A noble gas.", "No, a noble gas."),
    "Halogens": ("A halogen.", "No, a halogen."),
    "Metalloids": ("A metalloid.", "Partly; it is a metalloid."),
    "Alkali metals": ("An alkali metal.", "Yes."),
    "Alkaline earth metals": ("An alkaline earth metal.", "Yes."),
    "Transition metals": ("A transition metal.", "Yes."),
    "Poor metals": ("A post-transition metal.", "Yes."),
    "Lanthanides": ("A lanthanide.", "Yes."), "Actinides": ("An actinide.", "Yes."),
}
# symbols that are also gate stop words ("be", "in", "at", "i", "u") can't be
# told apart from filler, so they get no "what element is <symbol>" fact
SYMBOL_STOPWORDS = {"be", "in", "at", "i", "u"}
# measured physical data only: superheavies are predictions; At and Fr are estimates
MEASURED_MAX_Z, ESTIMATED = 96, {85, 87}

# ISO code for every country named in COUNTRY_CAPITALS or facts.tsv.
# Constituent countries of the UK share its currency and calling code.
COUNTRY_CODES = {
    "france": "FR", "germany": "DE", "italy": "IT", "spain": "ES", "portugal": "PT",
    "the united kingdom": "GB", "ireland": "IE", "the netherlands": "NL", "belgium": "BE",
    "switzerland": "CH", "austria": "AT", "poland": "PL", "sweden": "SE", "norway": "NO",
    "denmark": "DK", "finland": "FI", "greece": "GR", "turkey": "TR", "russia": "RU",
    "ukraine": "UA", "egypt": "EG", "nigeria": "NG", "kenya": "KE", "south africa": "ZA",
    "morocco": "MA", "china": "CN", "japan": "JP", "south korea": "KR", "india": "IN",
    "pakistan": "PK", "indonesia": "ID", "thailand": "TH", "vietnam": "VN",
    "the philippines": "PH", "saudi arabia": "SA", "iran": "IR", "israel": "IL",
    "australia": "AU", "new zealand": "NZ", "the united states": "US", "canada": "CA",
    "mexico": "MX", "brazil": "BR", "argentina": "AR", "chile": "CL", "peru": "PE",
    "colombia": "CO", "afghanistan": "AF", "albania": "AL", "algeria": "DZ", "andorra": "AD",
    "angola": "AO", "antigua and barbuda": "AG", "armenia": "AM", "azerbaijan": "AZ",
    "the bahamas": "BS", "bahrain": "BH", "bangladesh": "BD", "barbados": "BB", "belarus": "BY",
    "belize": "BZ", "benin": "BJ", "bhutan": "BT", "bolivia": "BO",
    "bosnia and herzegovina": "BA", "botswana": "BW", "brunei": "BN", "bulgaria": "BG",
    "burkina faso": "BF", "burundi": "BI", "cambodia": "KH", "cameroon": "CM",
    "cape verde": "CV", "the central african republic": "CF", "chad": "TD", "comoros": "KM",
    "the republic of the congo": "CG", "the democratic republic of the congo": "CD",
    "costa rica": "CR", "ivory coast": "CI", "croatia": "HR", "cuba": "CU", "cyprus": "CY",
    "the czech republic": "CZ", "djibouti": "DJ", "dominica": "DM",
    "the dominican republic": "DO", "ecuador": "EC", "el salvador": "SV",
    "equatorial guinea": "GQ", "eritrea": "ER", "estonia": "EE", "eswatini": "SZ",
    "ethiopia": "ET", "fiji": "FJ", "gabon": "GA", "the gambia": "GM", "georgia": "GE",
    "ghana": "GH", "grenada": "GD", "guatemala": "GT", "guinea": "GN", "guinea-bissau": "GW",
    "guyana": "GY", "haiti": "HT", "honduras": "HN", "hungary": "HU", "iceland": "IS",
    "iraq": "IQ", "jamaica": "JM", "jordan": "JO", "kazakhstan": "KZ", "kiribati": "KI",
    "kuwait": "KW", "kyrgyzstan": "KG", "laos": "LA", "latvia": "LV", "lebanon": "LB",
    "lesotho": "LS", "liberia": "LR", "libya": "LY", "liechtenstein": "LI", "lithuania": "LT",
    "luxembourg": "LU", "madagascar": "MG", "malawi": "MW", "malaysia": "MY",
    "the maldives": "MV", "mali": "ML", "malta": "MT", "the marshall islands": "MH",
    "mauritania": "MR", "mauritius": "MU", "micronesia": "FM", "moldova": "MD", "monaco": "MC",
    "mongolia": "MN", "montenegro": "ME", "mozambique": "MZ", "myanmar": "MM", "namibia": "NA",
    "nauru": "NR", "nepal": "NP", "nicaragua": "NI", "niger": "NE", "north korea": "KP",
    "north macedonia": "MK", "oman": "OM", "palau": "PW", "panama": "PA",
    "papua new guinea": "PG", "paraguay": "PY", "qatar": "QA", "romania": "RO", "rwanda": "RW",
    "saint kitts and nevis": "KN", "saint lucia": "LC", "saint vincent and the grenadines": "VC",
    "samoa": "WS", "san marino": "SM", "sao tome and principe": "ST", "senegal": "SN",
    "serbia": "RS", "seychelles": "SC", "sierra leone": "SL", "singapore": "SG",
    "slovakia": "SK", "slovenia": "SI", "the solomon islands": "SB", "somalia": "SO",
    "south sudan": "SS", "sri lanka": "LK", "sudan": "SD", "suriname": "SR", "syria": "SY",
    "taiwan": "TW", "tajikistan": "TJ", "tanzania": "TZ", "east timor": "TL", "togo": "TG",
    "tonga": "TO", "trinidad and tobago": "TT", "tunisia": "TN", "turkmenistan": "TM",
    "tuvalu": "TV", "uganda": "UG", "the united arab emirates": "AE", "uruguay": "UY",
    "uzbekistan": "UZ", "vanuatu": "VU", "venezuela": "VE", "yemen": "YE", "zambia": "ZM",
    "zimbabwe": "ZW", "vatican city": "VA", "kosovo": "XK", "greenland": "GL",
    "scotland": "GB", "wales": "GB", "northern ireland": "GB", "england": "GB",
}
COUNTRY_EXTRA = {"the united kingdom": ["the uk", "britain"],
                 "the united states": ["the usa", "the us", "america"]}

GAS_MARKS = {1: (140, 275), 2: (150, 300), 3: (170, 325), 4: (180, 350), 5: (190, 375),
             6: (200, 400), 7: (220, 425), 8: (230, 450), 9: (240, 475)}

CANADA_CAPITALS = {
    "alberta": "Edmonton.", "british columbia": "Victoria.", "manitoba": "Winnipeg.",
    "new brunswick": "Fredericton.", "newfoundland and labrador": "St. John's.",
    "nova scotia": "Halifax.", "ontario": "Toronto.", "prince edward island": "Charlottetown.",
    "quebec": "Quebec City.", "saskatchewan": "Regina.", "northwest territories": "Yellowknife.",
    "nunavut": "Iqaluit.", "yukon": "Whitehorse.",
}
# codes that read as gate stop words can't be told apart from filler
ABBREV_STOPWORDS = {"in", "or", "me", "hi", "ok", "de", "on", "so", "do", "be", "is", "it"}

CONSTANTS = {  # scipy.constants name: (phrasings, unit to print, significant digits)
    "Planck constant": (["what is the planck constant", "planck's constant value"], "J s", 9),
    "reduced Planck constant": (["what is the reduced planck constant", "what is h bar"], "J s", 7),
    "Boltzmann constant": (["what is the boltzmann constant", "boltzmann's constant"], "J/K", 7),
    "Avogadro constant": (["what is the avogadro constant", "avogadro number value"], "per mol", 9),
    "elementary charge": (["what is the elementary charge", "charge of a proton in coulombs"], "C", 10),
    "electron mass": (["what is the mass of an electron", "electron mass"], "kg", 6),
    "proton mass": (["what is the mass of a proton", "proton mass"], "kg", 6),
    "neutron mass": (["what is the mass of a neutron", "neutron mass"], "kg", 6),
    "molar gas constant": (["what is the gas constant", "ideal gas constant"], "J/(mol K)", 6),
    "Stefan-Boltzmann constant": (["what is the stefan-boltzmann constant",
                                   "stefan boltzmann constant"], "W/(m2 K4)", 5),
    "vacuum electric permittivity": (["what is the permittivity of free space",
                                      "what is epsilon zero"], "F/m", 6),
    "vacuum mag. permeability": (["what is the permeability of free space",
                                  "what is mu zero"], "N/A2", 6),
    "fine-structure constant": (["what is the fine structure constant",
                                 "fine-structure constant value"], "", 6),
    "Bohr radius": (["what is the bohr radius", "radius of a hydrogen atom"], "m", 6),
    "standard acceleration of gravity": (["what is standard gravity", "what is g in m/s2"],
                                         "m/s2", 6),
    "standard atmosphere": (["what is standard atmospheric pressure",
                             "how much is one atmosphere"], "Pa", 6),
    "Faraday constant": (["what is the faraday constant", "faraday's constant"], "C/mol", 6),
    "Rydberg constant": (["what is the rydberg constant", "rydberg constant value"], "1/m", 8),
    "Wien wavelength displacement law constant": (["what is wien's constant",
                                                    "wien displacement constant"], "m K", 5),
    "electron volt": (["how many joules in an electron volt", "what is an electron volt in joules"],
                      "J", 10),
}

BABY_ANIMALS = {
    "horse": "A foal.", "cow": "A calf.", "sheep": "A lamb.", "goat": "A kid.",
    "frog": "A tadpole.", "owl": "An owlet.", "swan": "A cygnet.", "bear": "A cub.",
    "deer": "A fawn.", "duck": "A duckling.", "goose": "A gosling.", "pig": "A piglet.",
    "rabbit": "A kit.", "fox": "A kit.", "elephant": "A calf.", "whale": "A calf.",
    "chicken": "A chick.", "eagle": "An eaglet.", "butterfly": "A caterpillar.",
    "lion": "A cub.", "tiger": "A cub.", "kangaroo": "A joey.", "cat": "A kitten.",
    "dog": "A puppy.",
}

ANIMAL_GROUPS = {
    "fish": "A school.", "geese": "A gaggle.", "cows": "A herd.", "bees": "A swarm.",
    "owls": "A parliament.", "ravens": "An unkindness.", "flamingos": "A flamboyance.",
    "dolphins": "A pod.", "whales": "A pod.", "sheep": "A flock.", "ants": "A colony.",
    "zebras": "A dazzle.", "gorillas": "A troop.", "hyenas": "A cackle.",
    "kangaroos": "A mob.",
}

WORKS = {  # title: (verb, creator)
    "don quixote": ("wrote", "Miguel de Cervantes."), "war and peace": ("wrote", "Leo Tolstoy."),
    "crime and punishment": ("wrote", "Fyodor Dostoevsky."),
    "the great gatsby": ("wrote", "F. Scott Fitzgerald."),
    "to kill a mockingbird": ("wrote", "Harper Lee."), "moby dick": ("wrote", "Herman Melville."),
    "harry potter": ("wrote", "J.K. Rowling."), "the lord of the rings": ("wrote", "J.R.R. Tolkien."),
    "the hobbit": ("wrote", "J.R.R. Tolkien."), "frankenstein": ("wrote", "Mary Shelley."),
    "dracula": ("wrote", "Bram Stoker."), "the divine comedy": ("wrote", "Dante Alighieri."),
    "on the origin of species": ("wrote", "Charles Darwin."),
    "a brief history of time": ("wrote", "Stephen Hawking."),
    "the art of war": ("wrote", "Sun Tzu."), "animal farm": ("wrote", "George Orwell."),
    "the little prince": ("wrote", "Antoine de Saint-Exupery."),
    "alice in wonderland": ("wrote", "Lewis Carroll."),
    "tom sawyer": ("wrote", "Mark Twain."), "sherlock holmes": ("wrote", "Arthur Conan Doyle."),
    "macbeth": ("wrote", "William Shakespeare."), "the iliad": ("wrote", "Homer."),
    "les miserables": ("wrote", "Victor Hugo."),
    "the communist manifesto": ("wrote", "Karl Marx and Friedrich Engels."),
    "the scream": ("painted", "Edvard Munch."), "the last supper": ("painted", "Leonardo da Vinci."),
    "guernica": ("painted", "Pablo Picasso."),
    "the persistence of memory": ("painted", "Salvador Dali."),
    "girl with a pearl earring": ("painted", "Johannes Vermeer."),
    "the birth of venus": ("painted", "Sandro Botticelli."),
    "the sistine chapel ceiling": ("painted", "Michelangelo."),
    "the night watch": ("painted", "Rembrandt."), "water lilies": ("painted", "Claude Monet."),
    "the creation of adam": ("painted", "Michelangelo."),
    "moonlight sonata": ("composed", "Ludwig van Beethoven."),
    "fur elise": ("composed", "Ludwig van Beethoven."),
    "the magic flute": ("composed", "Wolfgang Amadeus Mozart."),
    "the nutcracker": ("composed", "Pyotr Tchaikovsky."),
    "swan lake": ("composed", "Pyotr Tchaikovsky."),
    "clair de lune": ("composed", "Claude Debussy."),
    "the messiah": ("composed", "George Frideric Handel."),
    "canon in d": ("composed", "Johann Pachelbel."),
    "the blue danube": ("composed", "Johann Strauss II."),
}

INVENTIONS = {
    "the radio": "Guglielmo Marconi, with others.",
    "the television": "John Logie Baird and Philo Farnsworth.",
    "the steam engine": "Thomas Newcomen; James Watt improved it.",
    "the computer": "Charles Babbage, conceptually.",
    "electricity": "Nobody. It was discovered, not invented.",
    "the car": "Karl Benz, in 1886.", "the bicycle": "Karl Drais, in 1817.",
    "the internet": "Vint Cerf and Bob Kahn.", "email": "Ray Tomlinson.",
    "the battery": "Alessandro Volta.", "the lightning rod": "Benjamin Franklin.",
    "dynamite": "Alfred Nobel.", "the telescope": "Hans Lippershey, in 1608.",
    "the vaccine": "Edward Jenner.", "the periodic table": "Dmitri Mendeleev.",
    "alternating current": "Nikola Tesla, with George Westinghouse.",
    "the transistor": "Bardeen, Brattain and Shockley.",
    "the integrated circuit": "Jack Kilby and Robert Noyce.",
    "the computer mouse": "Douglas Engelbart.",
    "the printing press": "Johannes Gutenberg, around 1440.",
    "java": "James Gosling.", "javascript": "Brendan Eich.", "c++": "Bjarne Stroustrup.",
    "rust": "Graydon Hoare.", "unix": "Ken Thompson and Dennis Ritchie.",
    "git": "Linus Torvalds.", "the arduino": "Massimo Banzi and team, in 2005.",
}

PLANETS = {  # planet: (distance from sun, year length)
    "mercury": ("0.39 AU, about 58 million km.", "88 Earth days."),
    "venus": ("0.72 AU, about 108 million km.", "225 Earth days."),
    "mars": ("1.52 AU, about 228 million km.", None),
    "jupiter": ("5.2 AU, about 778 million km.", "11.9 Earth years."),
    "saturn": ("9.5 AU, about 1.4 billion km.", "29.5 Earth years."),
    "uranus": ("19.2 AU, about 2.9 billion km.", "84 Earth years."),
    "neptune": ("30.1 AU, about 4.5 billion km.", "165 Earth years."),
}


def ascii(text):
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


def sig(v, digits):
    """1.380649e-23 -> '1.380649 x 10^-23'. Plain decimals for ordinary sizes."""
    if 1e-3 <= abs(v) < 1e7:
        return f"{v:,.{digits}g}" if abs(v) < 1e5 else f"{v:,.0f}"
    m, e = f"{v:.{digits - 1}e}".split("e")
    return f"{m.rstrip('0').rstrip('.')} x 10^{int(e)}"


def celsius(kelvin):
    c = kelvin - 273.15
    return f"{c:,.1f} C." if abs(c) < 100 else f"{c:,.0f} C."


def emit_elements():
    from mendeleev import element

    print("# elements (mendeleev)")
    for z in range(1, 119):
        e = element(z)
        name, sym = e.name.lower(), e.symbol
        names = [name] + ELEMENT_ALIASES.get(name, [])
        each = lambda *templates: [t.format(n=n) for n in names for t in templates]
        emit(each("what is the chemical symbol for {n}", "what is the symbol for {n}",
                  "{n} symbol"), sym + ".")
        emit(each("what is the atomic number of {n}", "{n} atomic number",
                  "what number is {n} on the periodic table"), f"{z}.")
        if sym.lower() not in SYMBOL_STOPWORDS:
            emit([f"what element has the symbol {sym.lower()}", f"what element is {sym.lower()}"],
                 e.name + ".")
        emit([f"which element has atomic number {z}", f"what element is number {z}",
              f"element {z}"], e.name + ".")
        stable = z <= 83 and z not in (43, 61) or z in (90, 91, 92)
        mass = f"{e.atomic_weight:.3f} u." if stable else f"{e.atomic_weight:.0f} u, its most stable isotope."
        emit(each("what is the atomic mass of {n}", "what is the atomic weight of {n}",
                  "{n} atomic mass"), mass)
        if e.period:
            emit(each("what period is {n} in", "which period is {n} in", "{n} period"),
                 f"Period {e.period}.")
        if e.group_id:
            emit(each("what group is {n} in", "which group is {n} in", "{n} group"),
                 f"Group {e.group_id}.")
        if z <= 103 and e.series in SERIES:
            kind, metal = SERIES[e.series]
            emit(each("what type of element is {n}", "what kind of element is {n}",
                      "{n} element type"), kind)
            emit(each("is {n} a metal", "is {n} metal"), metal)
        if name in DISCOVERED or name in ANCIENT:
            who, when = DISCOVERED.get(name, ("Known since ancient times.", "In ancient times."))
            emit(each("who discovered {n}", "who found {n}"), who)
            emit(each("when was {n} discovered", "what year was {n} discovered"), when)
        if z > MEASURED_MAX_Z or z in ESTIMATED:
            continue
        mp, bp = e.melting_point, e.boiling_point
        if z == 33:  # arsenic sublimes at 1 atm
            mp = bp = None
        if mp is not None:
            emit(each("what is the melting point of {n}", "at what temperature does {n} melt",
                      "{n} melting point"), celsius(mp))
        if bp is not None and (mp is not None or z == 2):
            emit(each("what is the boiling point of {n}", "at what temperature does {n} boil",
                      "{n} boiling point"), celsius(bp))
        if e.density:
            d = e.density
            dens = f"{d * 1000:.3g} g/L." if d < 0.01 else f"{d:.3g} g/cm3."
            emit(each("what is the density of {n}", "how dense is {n}", "{n} density"), dens)
        if bp is not None and bp < 298.15:
            state = "A gas."
        elif mp is not None and mp <= 298.15:
            state = "A liquid."
        elif mp is not None or name in ("carbon", "phosphorus", "sulfur", "selenium"):
            state = "A solid."
        else:
            state = None
        if state:
            emit(each("is {n} a solid liquid or gas", "what state is {n} at room temperature"),
                 state)


def emit_countries():
    import phonenumbers
    from babel.numbers import get_currency_name, get_territory_currencies

    print("# countries (CLDR currencies, phonenumbers calling codes)")
    for country, code in COUNTRY_CODES.items():
        names = [country] + COUNTRY_EXTRA.get(country, [])
        if country.startswith("the "):
            names.append(country[4:])
        each = lambda *templates: [t.format(n=n) for n in names for t in templates]
        cur = get_territory_currencies(code, tender=True, non_tender=False)
        if cur:
            text = " and ".join(f"{ascii(get_currency_name(c, locale='en'))} ({c})" for c in cur)
            if len(text) < 48:
                emit(each("what is the currency of {n}", "what currency does {n} use",
                          "what money does {n} use", "{n} currency"), text + ".")
        cc = phonenumbers.country_code_for_region(code)
        if cc:
            emit(each("what is the calling code for {n}", "what is the dialing code for {n}",
                      "{n} phone code"), f"+{cc}.")
            emit(each("what is the country code for {n}"), f"+{cc} to call; {code} as an ISO code.")


def emit_subdivisions():
    import pycountry

    print("# US states and Canadian provinces (pycountry codes)")
    for country in ("US", "CA"):
        for sub in pycountry.subdivisions.get(country_code=country):
            if sub.type not in ("State", "District", "Province", "Territory"):
                continue
            name, ab = ascii(sub.name).lower(), sub.code.split("-")[1]
            if name == "district of columbia":
                names = [name, "washington dc", "dc"]
            else:
                names = [name]
            emit([t.format(n=n) for n in names for t in ("what is the abbreviation for {n}",
                  "{n} abbreviation", "what is the postal code for {n}")], ab + ".")
            if ab.lower() not in ABBREV_STOPWORDS:
                kind = "state" if country == "US" else "province"
                emit([f"what {kind} is {ab.lower()}", f"which {kind} has the abbreviation {ab.lower()}"],
                     ascii(sub.name) + ".")
    for prov, cap in CANADA_CAPITALS.items():
        emit([f"what is the capital of {prov}", f"capital of {prov}"], cap)


def emit_constants():
    import scipy.constants as sc

    print("# physical constants (CODATA via scipy)")
    for key, (questions, unit, digits) in CONSTANTS.items():
        value = sc.physical_constants[key][0]
        emit(questions, (sig(value, digits) + (" " + unit if unit else "")).strip() + ".")


def emit(questions, answer):
    print("|".join(questions) + "\t" + answer)


def main():
    print("# capitals")
    for country, cap in COUNTRY_CAPITALS.items():
        names = [country] + EXTRA_NAMES.get(country, [])
        qs = []
        for n in names:
            qs += [f"what is the capital of {n}", f"capital of {n}"]
        if country.startswith("the "):
            qs.append(f"capital of {country[4:]}")
        emit(qs, cap)
    for state, cap in US_STATE_CAPITALS.items():
        emit([f"what is the capital of {state}", f"capital of {state}",
              f"what is the state capital of {state}"], cap)
    emit_elements()
    emit_countries()
    emit_subdivisions()
    emit_constants()
    print("# animals")
    for animal, word in BABY_ANIMALS.items():
        emit([f"what is a baby {animal} called", f"what do you call a baby {animal}"], word)
    for animals, word in ANIMAL_GROUPS.items():
        emit([f"what is a group of {animals} called", f"what do you call a group of {animals}"], word)
    print("# works")
    for title, (verb, who) in WORKS.items():
        noun = {"wrote": "author", "painted": "painter", "composed": "composer"}[verb]
        emit([f"who {verb} {title}", f"who is the {noun} of {title}",
              f"who made {title}"], who)
    print("# inventions")
    for thing, who in INVENTIONS.items():
        emit([f"who invented {thing}", f"who created {thing}"], who)
    print("# oven gas marks")
    for mark, (c, f) in GAS_MARKS.items():
        emit([f"what is gas mark {mark} in celsius", f"gas mark {mark} in fahrenheit",
              f"gas mark {mark}", f"what temperature is gas mark {mark}"], f"{c} C ({f} F).")
    print("# planets")
    for planet, (dist, year) in PLANETS.items():
        emit([f"how far is {planet} from the sun", f"distance from the sun to {planet}",
              f"how far is {planet}"], dist)
        if year:
            emit([f"how long is a year on {planet}", f"how long does {planet} take to orbit the sun"],
                 year)


if __name__ == "__main__":
    main()
