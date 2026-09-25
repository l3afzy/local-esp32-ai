"""Generates the templated part of facts.tsv (capitals, elements, ...).

    python data/more_facts.py > data/facts_generated.tsv

Hand-written facts live in facts.tsv; templated ones are generated here so
every entry gets the same set of phrasings."""

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

ELEMENTS = [  # name, symbol, atomic number
    ("hydrogen", "H", 1), ("helium", "He", 2), ("lithium", "Li", 3), ("beryllium", "Be", 4),
    ("boron", "B", 5), ("carbon", "C", 6), ("nitrogen", "N", 7), ("oxygen", "O", 8),
    ("fluorine", "F", 9), ("neon", "Ne", 10), ("sodium", "Na", 11), ("magnesium", "Mg", 12),
    ("aluminum", "Al", 13), ("silicon", "Si", 14), ("phosphorus", "P", 15),
    ("sulfur", "S", 16), ("chlorine", "Cl", 17), ("argon", "Ar", 18), ("potassium", "K", 19),
    ("calcium", "Ca", 20), ("titanium", "Ti", 22), ("chromium", "Cr", 24),
    ("manganese", "Mn", 25), ("iron", "Fe", 26), ("cobalt", "Co", 27), ("nickel", "Ni", 28),
    ("copper", "Cu", 29), ("zinc", "Zn", 30), ("bromine", "Br", 35), ("krypton", "Kr", 36),
    ("silver", "Ag", 47), ("tin", "Sn", 50), ("iodine", "I", 53), ("xenon", "Xe", 54),
    ("tungsten", "W", 74), ("platinum", "Pt", 78), ("gold", "Au", 79), ("mercury", "Hg", 80),
    ("lead", "Pb", 82), ("radon", "Rn", 86), ("radium", "Ra", 88), ("uranium", "U", 92),
    ("plutonium", "Pu", 94),
]

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
    print("# elements")
    for name, sym, num in ELEMENTS:
        emit([f"what is the chemical symbol for {name}", f"what is the symbol for {name}",
              f"{name} symbol"], sym + ".")
        emit([f"what is the atomic number of {name}", f"{name} atomic number",
              f"what number is {name} on the periodic table"], f"{num}.")
        emit([f"what element has the symbol {sym.lower()}", f"what element is {sym.lower()}"],
             name.capitalize() + ".")
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
    print("# planets")
    for planet, (dist, year) in PLANETS.items():
        emit([f"how far is {planet} from the sun", f"distance from the sun to {planet}",
              f"how far is {planet}"], dist)
        if year:
            emit([f"how long is a year on {planet}", f"how long does {planet} take to orbit the sun"],
                 year)


if __name__ == "__main__":
    main()
