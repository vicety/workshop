di = {'<blank>': 'prázdné',
      '<unk>': 'neznámý',
      '<s>': 'začíná',
      '</s>': 'konec',
      'mak': 'makroekonomika',
      'slz': 'služby',
      'tur': 'cestovní ruch',
      'prg': 'Pragensie',
      'fin': 'finanční služby',
      'eur': 'evropská unie - zpravodajství',
      'dpr': 'Doprava',
      'met': 'počasí',
      'pol': 'Politika',
      'zak': 'kriminalita a zákon',
      'for': 'parlamenty a vlády',
      'sta': 'Stavebnictví a reality',
      'efm': 'Firmy',
      'spo': 'sports',
      'sko': 'Školství',
      'med': 'média a reklama',
      'mag': 'časopis výběr',
      'spl': 'životní styl',
      'odb': 'práce a odbory',
      'pit': 'Telekomunikace a informační technologie',
      'obo': 'Obchod',
      'aut': 'automobilový průmysl',
      'ekl': 'prostředí',
      'kul': 'Kultura',
      'zdr': 'zdravotní služby',
      'vat': 'věda a technika',
      'sop': 'Sociální problémy',
      'den': 'zprávy a plány',
      'bur': 'burzy',
      'bup': 'Currency exchanges',
      'tlk': 'Telekomunikace',
      'che': 'Chemické a farmaceutické průmysl',
      'bos': 'Čeština ze zahraničí',
      'buk': 'Komoditní burzy',
      'zem': 'Zemědělství',
      'ptr': 'Potravinářský',
      'nab': 'Náboženství',
      'ene': 'Energie',
      'fot': 'fotbal',
      'bua': 'burzy akciové'
      }

with open('save_data.test.dict', 'r') as f:
    lines = f.readlines()
    labels = [line.strip().split()[0] for line in lines]

with open('save_data.test.dict.trans.cz', 'w') as f:
    trans = map(lambda x: di[x] + '\n', labels)
    f.writelines(trans)