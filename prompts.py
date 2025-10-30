prompt_judge = """Determine the type of the question (simple or complex). Answer \"{{Simple}}\" or \"{{Complex}}\".

Examples:
Question: Which college was attended by the artist who headlined the John Mayer 2008 Summer Tour?
Answer: {{Complex}}

Question: what country is the grand bahama island in
Answer: {{Simple}}

Question: What sports team owned by The Madison Square Garden Company did Tyson Chandler play for?
Answer: {{Complex}}

Question: Who are the current senators from the state that partially contains the Upper Mississippi River?
Answer: {{Complex}}

Question: where did saki live
Answer: {{Simple}}

Question: what did einstein do
Answer: {{Simple}}

Question: People from the country that uses the Iranian Rial currency speak what languages?
Answer: {{Complex}}

Question: Which person that held a government position earliest did Baron de Montesquie influence?
Answer: {{Complex}}

Question: what are the most common religions in the united states
Answer: {{Simple}}

Question: what else did eli whitney invent
Answer: {{Simple}}

Question: Who trades with China and has a capital city called Khartoum?
Answer: {{Complex}}

Question: what to do today in birmingham uk
Answer: {{Simple}}

Question: Who held a government position after 06-08-2010 and was the prime minister of Japan in 2012?
Answer: {{Complex}}

Question: What team that Shaq play for that own the 2008 NBA Finals championship?
Answer: {{Complex}}

Question: where was toni braxton born and raised
Answer: {{Simple}}

Question: Which player on the Toronto Maple Leafs has testicular cancer?
Answer: {{Complex}}

Question: what other movies has liam hemsworth
Answer: {{Simple}}

Question: who will play clary in city of bones
Answer: {{Simple}}

Question: Which movie featuring Ian Somerhalder was released most recently?
Answer: {{Complex}}

Question: what kind of books does nicholas sparks wrote
Answer: {{Simple}}

Question: what did charles dickens believe in
Answer: {{Simple}}

Question: what movies did madonna appear in
Answer: {{Simple}}

Question: where was the roman colosseum located
Answer: {{Simple}}

Question: Where did the country whose capital is Brisbane get it name?
Answer: {{Complex}}

Question: What country borders South America and contains Bulawayo?
Answer: {{Complex}}

Question: Which colleges did Newt Gingrich attend with fewer than 7754 undergraduates?
Answer: {{Complex}}

Question: What is the type of government where Pyongyang is the capital?
Answer: {{Complex}}

Question: in which city was president kennedy killed
Answer: {{Simple}}

Question: Where is the largest population number where they speak Afrikaans?
Answer: {{Complex}}

Question: what is charlie sheen 's dad 's name
Answer: {{Simple}}

Question: Of the countries which Greece share borders with, which has the country capital of Tirana?
Answer: {{Complex}}

Question: Which person holds the position of president in the nation that uses the Costa Rican colon as its main currency?
Answer: {{Complex}}

Question: who plays mary jane in spiderman 2
Answer: {{Simple}}

Question: People from the country that contains Central Division, Fiji people speak what languages?
Answer: {{Complex}}

Question: what is nina dobrev nationality
Answer: {{Simple}}

Question: What movie with film character named Mr. Woodson did Tupac star in?
Answer: {{Complex}}

Question: what to see near sedona arizona
Answer: {{Simple}}

Question: where are the netherlands on a world map
Answer: {{Simple}}

Question: Which museum that was founded before 1893-09-16 is a good place to visit in downtown Chicago?
Answer: {{Complex}}

Question: what did william shakespeare do for a living
Answer: {{Simple}}

Question for you:
Question: {}
Answer:"""

prompt_judge_grailqa = """Determine the type of the question (simple or complex). Answer \"{{Simple}}\" or \"{{Complex}}\".

Examples:
Question: Which college was attended by the artist who headlined the John Mayer 2008 Summer Tour?
Answer: {{Complex}}

Question: victorian legislative assembly is part of what specific government office?
Answer: {{Simple}}

Question: What sports team owned by The Madison Square Garden Company did Tyson Chandler play for?
Answer: {{Complex}}

Question: Who are the current senators from the state that partially contains the Upper Mississippi River?
Answer: {{Complex}}

Question: where is the genomic locus of chromosome mt (human)?
Answer: {{Simple}}

Question: name the record producer that produced smile in your face?
Answer: {{Simple}}

Question: People from the country that uses the Iranian Rial currency speak what languages?
Answer: {{Complex}}

Question: Which person that held a government position earliest did Baron de Montesquie influence?
Answer: {{Complex}}

Question: Which is the publisher of Deadly Cults: The Crimes of True Believers?
Answer: {{Simple}}

Question: what else did eli whitney invent
Answer: {{Simple}}

Question: Who trades with China and has a capital city called Khartoum?
Answer: {{Complex}}

Question: what to do today in birmingham uk
Answer: {{Simple}}

Question: Who held a government position after 06-08-2010 and was the prime minister of Japan in 2012?
Answer: {{Complex}}

Question: What team that Shaq play for that own the 2008 NBA Finals championship?
Answer: {{Complex}}

Question: where was toni braxton born and raised
Answer: {{Simple}}

Question: Which player on the Toronto Maple Leafs has testicular cancer?
Answer: {{Complex}}

Question: what other movies has liam hemsworth
Answer: {{Simple}}

Question: the album freddie mercury tribute has what type of content?
Answer: {{Simple}}

Question: Which movie featuring Ian Somerhalder was released most recently?
Answer: {{Complex}}

Question: what kind of books does nicholas sparks wrote
Answer: {{Simple}}

Question: kindred spirits: asher b. durand and the american landscape was exhibited in which exhibit?
Answer: {{Simple}}

Question: which military unit size designation has the unit of 5th regiment indiana cavalry?
Answer: {{Simple}}

Question: what type of medical trials can only a males get?
Answer: {{Simple}}

Question: Where did the country whose capital is Brisbane get it name?
Answer: {{Complex}}

Question: What country borders South America and contains Bulawayo?
Answer: {{Complex}}

Question: Which colleges did Newt Gingrich attend with fewer than 7754 undergraduates?
Answer: {{Complex}}

Question: What is the type of government where Pyongyang is the capital?
Answer: {{Complex}}

Question: hip hop culture belongs to which genre?
Answer: {{Simple}}

Question: Where is the largest population number where they speak Afrikaans?
Answer: {{Complex}}

Question: what is charlie sheen 's dad 's name
Answer: {{Simple}}

Question: Of the countries which Greece share borders with, which has the country capital of Tirana?
Answer: {{Complex}}

Question: Which person holds the position of president in the nation that uses the Costa Rican colon as its main currency?
Answer: {{Complex}}

Question: who plays mary jane in spiderman 2
Answer: {{Simple}}

Question: People from the country that contains Central Division, Fiji people speak what languages?
Answer: {{Complex}}

Question: zimbali golf course was designed by which golf course architect?
Answer: {{Simple}}

Question: What movie with film character named Mr. Woodson did Tupac star in?
Answer: {{Complex}}

Question: elton john aids foundation is partners with what organizations?
Answer: {{Simple}}

Question: where are the netherlands on a world map
Answer: {{Simple}}

Question: Which museum that was founded before 1893-09-16 is a good place to visit in downtown Chicago?
Answer: {{Complex}}

Question: what did william shakespeare do for a living
Answer: {{Simple}}

Question for you:
Question: {}
Answer:"""


prompt_answer_tri = """Reason and answer the question in \"{{answer entity}}\" according to the knowledge triplets and your knowledge.
Question: who was the wife of king edward vii
Knowledge Triplets: < Edward VII > [SEP] < common.topic.image > [SEP] < Edward VII and Queen Alexandra.jpg >
< Edward VII > [SEP] < people.person.children > [SEP] < Princess Victoria of the United Kingdom >
< m.02kp9gy > [SEP] < people.marriage.spouse > [SEP] < Edward VII >
< Edward VII > [SEP] < people.person.spouse_s > [SEP] < m.02kp9gy >
< m.02kp9gy > [SEP] < people.marriage.spouse > [SEP] < Alexandra of Denmark >
Answer: Based on the given knowledge triples, the wife of King Edward VII was Alexandra of Denmark. So the answer is {{Alexandra of Denmark}}.

Question: what position did john stockton play
Knowledge Triplets: < John Stockton > [SEP] < basketball.basketball_player.position_s > [SEP] < Point guard >
< Point guard > [SEP] < basketball.basketball_position.players > [SEP] < John Stockton >
< John Stockton > [SEP] < people.person.profession > [SEP] < Basketball player >
< John Stockton > [SEP] < sports.pro_athlete.sports_played_professionally > [SEP] < m.0d5jgl6 >
< John Stockton > [SEP] < basketball.basketball_player.player_statistics > [SEP] < m.04qqfwf >
Answer: Based on the given knowledge triples, John Stockton played the position of Point guard in basketball. So the answer is {{Point guard}}.

Question: where is tom cruise from
Knowledge Triplets: < Tom Cruise > [SEP] < people.person.nationality > [SEP] < United States of America >
< Syracuse > [SEP] < location.location.people_born_here > [SEP] < Tom Cruise >
< Tom Cruise > [SEP] < people.person.ethnicity > [SEP] < Irish American >
< Tom Cruise > [SEP] < people.person.ethnicity > [SEP] < English American >
< Tom Cruise > [SEP] < people.person.place_of_birth > [SEP] < Syracuse >
Answer: Based on the given knowledge triples, Tom Cruise was born in Syracuse, United States. Therefore, Tom Cruise is from Syracuse. So the answer is {{Syracuse}}. 

Question: who controls panama
Knowledge Triplets: < Panama Canal Authority > [SEP] < government.government_agency.jurisdiction > [SEP] < Panama >
< Panama > [SEP] < government.governmental_jurisdiction.agencies > [SEP] < Panama Canal Authority >
< Panama > [SEP] < government.governmental_jurisdiction.government_positions > [SEP] < Vice President of Panama >
< Panama > [SEP] < government.governmental_jurisdiction.government_positions > [SEP] < President of Panama >
< President of Panama > [SEP] < government.government_office_or_title.jurisdiction > [SEP] < Panama >
Answer: The given triples don't contain the answer to the question. But based on my knowledge, the President of Panama is Juan Carlos Varela. So the answer is {{Juan Carlos Varela}}. 

Question: {}
Knowledge triplets: {}
Answer:"""

prompt_answer_tri_grailqa = """Reason and answer the question in \"{{answer entity}}\" according to the given retrieved knowledge and your knowledge. The format of given knowledge is \"< topic_entity > [SEP] < 1_hop_relation > [SEP] [ 1_hop_entities ]\" or \"< topic_entity > [SEP] < 1_hop_relation > [SEP] < 2_hop_relation > [SEP] [ 2_hop_entities ]\" or the same structure in reverse order.
Question: name the record producer that produced smile in your face?
Retrieved knowledge: < Smile in Your Face > [SEP] < music.recording.producer > [SEP] < John Primer >
< John Primer > [SEP] < music.producer.tracks_produced > [SEP] < Smile in Your Face >
< Smile in Your Face > [SEP] < music.recording.producer > [SEP] < music.producer.tracks_produced > [SEP] < Yak Yak >
< Smile in Your Face > [SEP] < music.recording.producer > [SEP] < m.0132_q73 >
< m.0132_q73 > [SEP] < music.producer.tracks_produced > [SEP] < Smile in Your Face >
Answer: Based on the given knowledge, the record producer of Smile in Your Face are John Primer and "m.0132_q73". So the answers are {{John Primer}} and {{m.0132_q73}}.

Question: where is the genomic locus of chromosome mt (human)?
Retrieved knowledge: < MT + [5904,7445] > [SEP] < biology.genomic_locus.chromosome > [SEP] < Chromosome MT (human) >
< MT + [9207,9987] > [SEP] < biology.genomic_locus.chromosome > [SEP] < Chromosome MT (human) >
< MT + [8366,8572] > [SEP] < biology.genomic_locus.chromosome > [SEP] < Chromosome MT (human) >
< Chromosome MT (human) > [SEP] < biology.chromosome.locus > [SEP] < MT + [8366,8572] >
< MT + [7586,8269] > [SEP] < biology.genomic_locus.chromosome > [SEP] < Chromosome MT (human) >
Answer: Based on the given knowledge, the genomic locus of Chromosome MT (human) include MT + [5904,7445], MT + [9207,9987], MT + [8366,8572], and MT + [7586,8269]. So the answers are {{MT + [5904,7445]}}, {{MT + [9207,9987]}}, {{MT + [8366,8572]}}, and {{MT + [7586,8269]}}.

Question: victorian legislative assembly is part of what specific government office?
Retrieved knowledge: < Member of the Victorian Legislative Assembly > [SEP] < government.government_office_or_title.governmental_body_if_any > [SEP] < Victorian Legislative Assembly >
< Victorian Legislative Assembly > [SEP] < government.governmental_body.offices_positions > [SEP] < Member of the Victorian Legislative Assembly >
< Speaker of the Victorian Legislative Assembly > [SEP] < government.government_office_or_title.governmental_body_if_any > [SEP] < Victorian Legislative Assembly >
< Victorian Legislative Assembly > [SEP] < government.governmental_body.body_this_is_a_component_of > [SEP] < Parliament of Victoria >
< Victorian Legislative Assembly > [SEP] < government.governmental_body.offices_positions > [SEP] < Speaker of the Victorian Legislative Assembly >
Answer: Based on the given knowledge, Victorian Legislative Assembly has associated office positions, which are Member of the Victorian Legislative Assembly, and Speaker of the Victorian Legislative Assembly. Therefore, Victorian Legislative Assembly is part of Member of the Victorian Legislative Assembly and Speaker of the Victorian Legislative Assembly. So the answers are {{Member of the Victorian Legislative Assembly}} and {{Speaker of the Victorian Legislative Assembly}}. 

Question: Which is the publisher of Deadly Cults: The Crimes of True Believers?
Retrieved knowledge: < Deadly Cults: The Crimes of True Believers > [SEP] < book.book_edition.contributing_authors > [SEP] < Robert L. Snow >
< Robert L. Snow > [SEP] < book.author.contributing_author_to > [SEP] < Deadly Cults: The Crimes of True Believers >
< Deadly Cults > [SEP] < book.written_work.author > [SEP] < Robert L. Snow >
< Robert L. Snow > [SEP] < book.author.works_written > [SEP] < Deadly Cults >
< Robert L. Snow > [SEP] < book.author.works_written > [SEP] < Sex crimes investigation >
Answer: I can't deduce the answer from the given knowledge. But based on my knowledge, the publisher of Deadly Cults: The Crimes of True Believers is Praeger Publishers. So the answer is {{Praeger Publishers}}. 

Question: the album freddie mercury tribute has what type of content?
Retrieved knowledge: < Freddie Mercury Tribute > [SEP] < music.album.album_content_type > [SEP] < Compilation album >
< Freddie Mercury Tribute > [SEP] < music.album.album_content_type > [SEP] < Live Album >
< Compilation album > [SEP] < music.album_content_type.albums > [SEP] < Freddie Mercury Tribute >
< Live Album > [SEP] < music.album_content_type.albums > [SEP] < Freddie Mercury Tribute >
< Freddie Mercury Tribute > [SEP] < common.topic.notable_types > [SEP] < Live Album >
Answer: Based on the given knowledge, the album Freddie Mercury Tribute is a Compilation album and a Live Album. So the answers are {{Compilation album}} and {{Live Album}}.

Question: {}
Retrieved knowledge: {}
Answer:"""

prompt_answer_rel = """Reason and answer the question in \"{{answer entity}}\" according to the given retrieved knowledge and your knowledge. The format of given knowledge is \"< topic_entity > [SEP] < 1_hop_relation > [SEP] [ 1_hop_entities ]\" or \"< topic_entity > [SEP] < 1_hop_relation > [SEP] < 2_hop_relation > [SEP] [ 2_hop_entities ]\" or the same structure in reverse order.
Question: who was the wife of king edward vii
Retrieved knowledge: < Edward VII > [SEP] < people.person.children > [SEP] [ < Princess Victoria of the United Kingdom > < Maud of Wales > < Prince Albert Victor, Duke of Clarence and Avondale > < Louise, Princess Royal > < Prince Alexander John of Wales > < George V > ]
[ < United Kingdom > < British Raj > ] [SEP] < royalty.kingdom.rulers > [SEP] < Edward VII >
< Edward VII > [SEP] < royalty.noble_person.titles > [SEP] [ < m.04l81l3 > < m.05yx_1l > < m.05cwn46 > ]
< Edward VII > [SEP] < people.person.spouse_s > [SEP] < people.marriage.spouse > [SEP] [ < Alexandra of Denmark > < Edward VII > ]
< Edward VII > [SEP] < people.person.parents > [SEP] [ < Queen Victoria > < Albert, Prince Consort > ]
Answer: Based on the given knowledge, the wife of King Edward VII was Alexandra of Denmark. So the answer is {{Alexandra of Denmark}}.

Question: what position did john stockton play
Retrieved knowledge: [ < Point guard > ] [SEP] < basketball.basketball_position.players > [SEP] < John Stockton >
< John Stockton > [SEP] < basketball.basketball_player.position_s > [SEP] [ < Point guard > ]
[ < Point guard > ] [SEP] < sports.sports_position.players > [SEP] < sports.sports_team_roster.player > [SEP] < John Stockton >
< John Stockton > [SEP] < sports.pro_athlete.sports_played_professionally > [SEP] [ < m.0d5jgl6 > ]
< John Stockton > [SEP] < basketball.basketball_player.player_statistics > [SEP] [ < m.04ql4pb > < m.04qpgqk > < m.04qs7_2 > < m.04qj834 > < m.04qgggm > < m.04qdshs > < m.04qk_mj > < m.04qrys0 > < m.04qpzrp > < m.04qsbz1 > ]
Answer: Based on the given knowledge, John Stockton played the position of Point guard in basketball. So the answer is {{Point guard}}.

Question: where is tom cruise from
Retrieved knowledge:  [ < Syracuse > ] [SEP] < location.location.people_born_here > [SEP] < Tom Cruise >         
< Tom Cruise > [SEP] < people.person.nationality > [SEP] [ < United States of America > ]               
< Tom Cruise > [SEP] < people.person.places_lived > [SEP] [ < m.0c1wjv8 > < m.03pncvj > < m.03pj6v9 > < m.03pkdcv > < m.0gx9rhk > ]                                                                             
< Tom Cruise > [SEP] < people.person.place_of_birth > [SEP] [ < Syracuse > ]                            
< Tom Cruise > [SEP] < base.popstra.celebrity.vacations_in > [SEP] < base.popstra.vacation_choice.location > [SEP] [ < Sardinia > < Las Vegas > < Paloma Beach in Saint-Jean Cap Ferrat > < London > < Tahiti > < Fiji > < Maldives > < Paris > < Telluride > < Saint-Tropez > ]
Answer: Based on the given knowledge, Tom Cruise was born in Syracuse, United States. Therefore, Tom Cruise is from Syracuse. So the answer is {{Syracuse}}. 

Question: who controls panama
Retrieved knowledge: < Panama > [SEP] < government.governmental_jurisdiction.governing_officials > [SEP] [ < m.010ghc3_ > ]
< Panama > [SEP] < government.governmental_jurisdiction.governing_officials > [SEP] < government.government_position_held.office_holder > [SEP] [ < Felipe Alejandro Virzi Lopez > < Juan Carlos Varela > < Ricardo Martinelli > < Martín Torrijos > < Ernesto Pérez Balladares > < Samuel Lewis Navarro > < Manuel Solís Palma > < Ricardo Arias Calderón > < Tomas Altamirano Duque > < Mireya Moscoso > < Isabel Saint Malo > < Arturo Vallarino > < Guillermo Ford > < Dominador Baldomero Bazán > < Jorge Illueca > < Nicolás Ardito Barletta Vallarino > < Rubén Arosemena > ]
< Panama > [SEP] < government.governmental_jurisdiction.governing_officials > [SEP] < government.government_position_held.to > [SEP] [ < 1999 > < 2014-07-01 > < 2009-07-01 > < 1999-09-01 > < 1989-09-01 > < 1992-12-17 > < 2004-09-01 > < 2004-08-30 > < 1992 > < 1994 > < 2004 > < 1984-10-11 > < 1985-09-28 > < 2009 > ]
< Panama > [SEP] < government.governmental_jurisdiction.governing_officials > [SEP] < government.government_position_held.from > [SEP] [ < 1994 > < 2009-07-01 > < 2014-07-01 > < 2004-09-01 > < 1994-09-01 > < 1988-02-26 > < 1990-12 > < 1999-09-01 > < 1989 > < 1992 > < 1999 > < 1984-02-13 > < 1984-10-11 > ]
< Panama > [SEP] < government.governmental_jurisdiction.governing_officials > [SEP] < government.government_position_held.jurisdiction_of_office > [SEP] [ < Panama > ]
Answer: I can't deduce the answer from the given knowledge. But based on my knowledge, the President of Panama is Juan Carlos Varela. So the answer is {{Juan Carlos Varela}}. 

Question: what happened to nagasaki
Retrieved knowledge: [ < Iccho Itoh > ] [SEP] < user.alexander.misc.murdered_person.place_murdered > [SEP] < Nagasaki >
< Nagasaki > [SEP] < location.location.events > [SEP] [ < Atomic bombings of Hiroshima and Nagasaki > < Shinee World 2014 > < Miss International 1986 > ]
[ < Atomic bombings of Hiroshima and Nagasaki > < Shinee World 2014 > < Miss International 1986 > ] [SEP] < time.event.locations > [SEP] < Nagasaki >
< Nagasaki > [SEP] < location.location.contains > [SEP] [ < Nagasaki University > < Kwassui Women's University > < Nagasaki Institute of Applied Science > < Nagasaki Peace Park > < Mount Inasa > < Nagasaki Junshin Catholic University > < Fukuoka #2 > < Nagasaki International University > < Nagasaki Museum of History and Culture > ]
[ < Antonio Gonzalez > < Lorenzo Ruiz > < William Robert McFarlane > < Masaharu Taniguchi > < Magdalene of Nagasaki > < Philip of Jesus > < Michael de Aozaraza > < Carlo Spinola > < Iccho Itoh > < Paulo Miki > < Gonsalo Garcia > < Tsutomu Yamaguchi > < Itsunen Shoyu > < Einosuke Harada > < Joseph Asajiro Satowaki > < Jacobo Kyushei Tomonaga > < Domingo Ibáñez de Erquicia > < Caius of Korea > ] [SEP] < people.deceased_person.place_of_death > [SEP] < Nagasaki >
Answer: Based on the given knowledge, Nagasaki was the location of the Atomic bombings of Hiroshima and Nagasaki, as well as events like Shinee World 2014 and Miss International 1986. So the answers are {{Atomic bombings of Hiroshima and Nagasaki}}, {{Shinee World 2014}}, and {{Miss International 1986}}.

Question: {}
Retrieved knowledge: {}
Answer:"""

prompt_answer_rel_grailqa = """Reason and answer the question in \"{{answer entity}}\" according to the given retrieved knowledge and your knowledge. The format of given knowledge is \"< topic_entity > [SEP] < 1_hop_relation > [SEP] [ 1_hop_entities ]\" or \"< topic_entity > [SEP] < 1_hop_relation > [SEP] < 2_hop_relation > [SEP] [ 2_hop_entities ]\" or the same structure in reverse order.
Question: name the record producer that produced smile in your face?
Retrieved knowledge: < Smile in Your Face > [SEP] < music.recording.producer > [SEP] < music.producer.tracks_produced > [SEP] [ < I'm Just A Fool > < Doing My Own Thing/I Can't Get Next to You/I'm a Bluesman > < Empty Arms > < I Should Have Quit You > < When You Got a Headache > < 18-39 > < Maggie > < Blues With a Felling > < You Better Watch Yourself > < I'm Doing Too Bad > < Wonder What's the Matter > < Good Morning Heartache > < Corrine > < Rough Dried Woman > < Need Some Good Lovin' > < Baby Can't You See > < Got to Be Some Changes > < Dry in the Dark > < Bump and Grind > < Every Time You Leave Me > ]
[ < Corrine > < I Got What It Takes > < Sittin' in My Bedroom > < Sometimes I Wonder > < Yak Yak > < My Eyes Keep Me in Trouble > < I'm Hooked on Your Love > < I'm Doing Too Bad > < Doing My Own Thing/I Can't Get Next to You/I'm a Bluesman > < Rough Dried Woman > < Yes, I'm Crazy About My Baby > < Change My Mind > < Highway Is My Home > < Blues With a Felling > < I Got to Find My Baby > < 18-39 > < I'm Good > < Little Bluebird > < I'm Tore Upo > < Next Time You See Me > ] [SEP] < music.recording.producer > [SEP] < music.producer.tracks_produced > [SEP] < Smile in Your Face >
< Smile in Your Face > [SEP] < music.recording.producer > [SEP] [ < John Primer > < m.0132_q73 > ]
< Smile in Your Face > [SEP] < music.recording.producer > [SEP] < common.topic.notable_for > [SEP] [ < g.11b7nymlkq > ]
[ < John Primer > < m.0132_q73 > ] [SEP] < music.producer.tracks_produced > [SEP] < Smile in Your Face >
Answer: Based on the given knowledge, the record producer of Smile in Your Face are John Primer and "m.0132_q73". So the answers are {{John Primer}} and {{m.0132_q73}}.

Question: where is the genomic locus of chromosome mt (human)?
Retrieved knowledge: [ < MT + [3307,4263] > < MT + [9207,9987] > < MT + [4470,5511] > < MT + [8366,8572] > < MT + [10470,10766] > < MT + [12337,14148] > < MT + [10059,10404] > < MT + [8527,9207] > < MT + [5904,7445] > < MT + [7586,8269] > < MT + [10760,12137] > < MT - [14149,14673] > < MT + [14747,15881] > ] [SEP] < biology.genomic_locus.chromosome > [SEP] < Chromosome MT (human) >
< Chromosome MT (human) > [SEP] < biology.chromosome.locus > [SEP] [ < MT + [3307,4263] > < MT + [9207,9987] > < MT + [4470,5511] > < MT + [8366,8572] > < MT + [10470,10766] > < MT + [12337,14148] > < MT + [10059,10404] > < MT + [8527,9207] > < MT + [5904,7445] > < MT + [7586,8269] > < MT + [10760,12137] > < MT - [14149,14673] > < MT + [14747,15881] > ]
< Chromosome MT (human) > [SEP] < biology.chromosome.genome > [SEP] [ < Human genome > ]
< Chromosome MT (human) > [SEP] < biology.chromosome.gene > [SEP] [ < mitochondrially encoded NADH 4L > < MT-ND1 > < mitochondrially encoded cytochrome c oxidase II > < mitochondrially encoded NADH dehydrogenase 6 > < mitochondrially encoded cytochrome c oxidase III > < mitochondrially encoded NADH dehydrogenase 2 > < mitochondrially encoded ATP synthase 8 > < mitochondrially encoded NADH dehydrogenase 4 > < mitochondrially encoded NADH dehydrogenase 5 > < mitochondrially encoded NADH dehydrogenase 3 > < mitochondrially encoded cytochrome c oxidase I > < mitochondrially encoded ATP synthase 6 > < mitochondrially encoded cytochrome b > ]
[ < Human genome > ] [SEP] < biology.genome.chromosome > [SEP] < Chromosome MT (human) >
Answer: Based on the given knowledge, the genomic locus of Chromosome MT (human) include MT + [3307,4263], MT + [9207,9987], MT + [4470,5511], MT + [8366,8572], MT + [10470,10766], MT + [12337,14148], MT + [10059,10404], MT + [8527,9207], MT + [5904,7445], MT + [7586,8269], MT + [10760,12137], MT - [14149,14673], and MT + [14747,15881]. So the answers are {{MT + [3307,4263]}}, {{MT + [9207,9987]}}, {{MT + [4470,5511]}}, {{MT + [8366,8572]}}, {{MT + [10470,10766]}}, {{MT + [12337,14148]}}, {{MT + [10059,10404]}}, {{MT + [8527,9207]}}, {{MT + [5904,7445]}}, {{MT + [7586,8269]}}, {{MT + [10760,12137]}}, {{MT - [14149,14673]}}, and {{MT + [14747,15881]}}.

Question: victorian legislative assembly is part of what specific government office?
Retrieved knowledge:  < Victorian Legislative Assembly > [SEP] < government.governmental_body.offices_positions > [SEP] [ < Member of the Victorian Legislative Assembly > < Speaker of the Victorian Legislative Assembly > ]
< Victorian Legislative Assembly > [SEP] < government.governmental_body.members > [SEP] < government.government_position_held.office_holder > [SEP] [ < Ray Buckley > < Arthur Robinson > < Richard Abbott > < Ted Baillieu > < Agar Wynne > < John Foster > < Robert Menzies > < Ben Carroll > < John Goodman > < Tom Corrigan > < John Dane > < Matthew Guy > < Jan Wilson > < David Cunningham > < Denis Napthine > < Michael O'Grady > < Paul Jenkins > < Rupert Hamer > < Jennifer Kanis > < Robert Doyle > ]
< Victorian Legislative Assembly > [SEP] < government.governmental_body.body_this_is_a_component_of > [SEP] [ < Parliament of Victoria > ]
[ < Parliament of Victoria > ] [SEP] < government.governmental_body.component_bodies > [SEP] < Victorian Legislative Assembly >
[ < Member of the Victorian Legislative Assembly > < Speaker of the Victorian Legislative Assembly > ] [SEP] < government.government_office_or_title.governmental_body_if_any > [SEP] < Victorian Legislative Assembly >
Answer: Based on the given knowledge, Victorian Legislative Assembly has associated office positions, which are Member of the Victorian Legislative Assembly, and Speaker of the Victorian Legislative Assembly. Therefore, Victorian Legislative Assembly is part of Member of the Victorian Legislative Assembly and Speaker of the Victorian Legislative Assembly. So the answers are {{Member of the Victorian Legislative Assembly}} and {{Speaker of the Victorian Legislative Assembly}}. 

Question: Which is the publisher of Deadly Cults: The Crimes of True Believers?
Retrieved knowledge: < Robert L. Snow > [SEP] < book.author.openlibrary_id > [SEP] [ < OL22272A > ]
[ < Deadly Cults: The Crimes of True Believers > ] [SEP] < book.book_edition.contributing_authors > [SEP] < Robert L. Snow >
[ < Looking for Carroll Beckwith > < Stopping a stalker > < Deadly Cults > < Sex crimes investigation > < The complete guide to personal and home safety > < Protecting your life, home, and property > < The militia threat > < Terrorists among us > < Technology and law enforcement > ] [SEP] < book.written_work.author > [SEP] < Robert L. Snow >
< Robert L. Snow > [SEP] < book.author.contributing_author_to > [SEP] [ < Deadly Cults: The Crimes of True Believers > ]
< Robert L. Snow > [SEP] < book.author.works_written > [SEP] [ < Looking for Carroll Beckwith > < Stopping a stalker > < Deadly Cults > < Sex crimes investigation > < The complete guide to personal and home safety > < Protecting your life, home, and property > < The militia threat > < Terrorists among us > < Technology and law enforcement > ]
Answer: I can't deduce the answer from the given knowledge. But based on my knowledge, the publisher of Deadly Cults: The Crimes of True Believers is Praeger Publishers. So the answer is {{Praeger Publishers}}. 

Question: the album freddie mercury tribute has what type of content?
Retrieved knowledge: [ < Compilation album > < Live Album > ] [SEP] < music.album_content_type.albums > [SEP] < Freddie Mercury Tribute >
< Freddie Mercury Tribute > [SEP] < music.album.album_content_type > [SEP] [ < Compilation album > < Live Album > ]
< Freddie Mercury Tribute > [SEP] < music.album.release_type > [SEP] [ < Album > ]
[ < Album > ] [SEP] < music.album_release_type.albums > [SEP] < Freddie Mercury Tribute >
[ < Queen > ] [SEP] < music.artist.album > [SEP] < Freddie Mercury Tribute >
Answer: Based on the given knowledge, the album Freddie Mercury Tribute is a Compilation album and a Live Album. So the answers are {{Compilation album}} and {{Live Album}}.

Question: {}
Retrieved knowledge: {}
Answer:"""

prompt_answer_gnn = """Reason and answer the question in \"{{answer entity}}\" according to the knowledge triplets and your knowledge.
Question: who won the governor election in puerto rico
Knowledge Triplets: < Puerto Rico > [SEP] < government.governmental_jurisdiction.governing_officials > [SEP] < government.government_position_held.office_holder > [SEP] < Luis Fortuño >
< Puerto Rico > [SEP] < government.governmental_jurisdiction.governing_officials > [SEP] < government.government_position_held.office_holder > [SEP] < Alejandro García Padilla >
< Puerto Rico > [SEP] < government.governmental_jurisdiction.governing_officials > [SEP] < government.government_position_held.basic_title > [SEP] < Governor >
Answer: Based on the given knowledge triples, the governor of Puerto Rico are Luis Fortuño and Alejandro García Padilla. So the answers are {{Luis Fortuño}} and {{Alejandro García Padilla}}.

Question: what did barack obama do before he took office
Knowledge Triplets: < Barack Obama > [SEP] < government.us_president.vice_president > [SEP] < Joe Biden >
< Barack Obama > [SEP] < people.person.profession > [SEP] < Lawyer >
< Barack Obama > [SEP] < people.person.profession > [SEP] < Writer >
< Barack Obama > [SEP] < people.person.profession > [SEP] < Politician >
< Barack Obama > [SEP] < people.person.profession > [SEP] < Law professor >
Answer: The given triples don't contain the answer to the question. The answer needs to be specific political roles that Barack Obama held, rather than just general classifications. But based on my knowledge, Barack Obama was a United States Senator and an Illinois State Senator. So the answers are {{United States Senator}} and {{Illinois State Senator}}.

Question: where is tom cruise from
Knowledge Triplets: < Tom Cruise > [SEP] < people.person.nationality > [SEP] < United States of America >
< Syracuse > [SEP] < location.location.people_born_here > [SEP] < Tom Cruise >
< Tom Cruise > [SEP] < people.person.ethnicity > [SEP] < Irish American >
< Tom Cruise > [SEP] < people.person.ethnicity > [SEP] < English American >
< Tom Cruise > [SEP] < people.person.place_of_birth > [SEP] < Syracuse >
Answer: Based on the given knowledge triples, Tom Cruise was born in Syracuse, United States. Therefore, Tom Cruise is from Syracuse. So the answer is {{Syracuse}}. 

Question: what are the museums located in vienna, austria
Knowledge Triples: < Vienna > [SEP] < travel.travel_destination.tourist_attractions > [SEP] < Kunsthistorisches Museum >
< Vienna > [SEP] < travel.travel_destination.tourist_attractions > [SEP] < St. Peter's Church >
< Vienna > [SEP] < travel.travel_destination.tourist_attractions > [SEP] < Millennium Tower >
< Vienna > [SEP] < travel.travel_destination.tourist_attractions > [SEP] < Leopold Museum >
< Vienna > [SEP] < travel.travel_destination.tourist_attractions > [SEP] < St. Stephen's Cathedral, Vienna >
< Vienna > [SEP] < travel.travel_destination.tourist_attractions > [SEP] < Liechtenstein Museum >
< Vienna > [SEP] < travel.travel_destination.tourist_attractions > [SEP] < Museum of Military History, Vienna >
< Vienna > [SEP] < travel.travel_destination.tourist_attractions > [SEP] < Hofburg Palace >
< Vienna > [SEP] < travel.travel_destination.tourist_attractions > [SEP] < Albertina >
< Vienna > [SEP] < travel.travel_destination.tourist_attractions > [SEP] < Naturhistorisches Museum >
Answer: Based on the given knowledge triples, Based on the given knowledge triplets, some of the museums located in Vienna, Austria include: Kunsthistorisches Museum, Leopold Museum, Liechtenstein Museum, Museum of Military History, Naturhistorisches Museum. So the answers are {{Kunsthistorisches Museum}}, {{Leopold Museum}}, {{Liechtenstein Museum}}, {{Museum of Military History}} and {{Naturhistorisches Museum}}.

Question: {}
Knowledge triplets: {}
Answer:"""


prompt_answer_extract = """Extract the answer entity of the question in form of \"{{answer entity}}\" according to the given text.
Question: what city was leonardo da vinci from
Text: The answer is Vinci. Based on the given knowledge triplets, Leonardo da Vinci was born in Vinci, Italy.
Answer: {{Vinci}}

Question: what does michael vick do
Text: Albert Einstein was a multifaceted individual whose contributions spanned various fields. As a theoretical physicist, he is best known for his groundbreaking work in developing the theory of relativity. Beyond his scientific achievements, Einstein was also recognized as a prominent mathematician, using mathematical rigor to develop and articulate his theories. Throughout his life, Einstein served as a teacher, sharing his knowledge and insights with students and colleagues. 
Answer: {{theoretical physicist}}, {{mathematician}}, {{teacher}}

Question: {}
Text: {}
Answer:"""


prompt_subclassify = """Based on the following examples, learn how to determine the type of a question based on its semantics, rather than relying on keywords in the question. Then, determine the type of the given question based on its semantics. Answer {}.

Examples:
{}
Question for you:
Question: {}
Answer:"""

prompt_subclassify_examples = {
    "composition": """Question: Who is the current leader of the place where the newspaper Granma is circulated?
Answer: The solution to this question requires first identifying the place where the Granma newspaper is circulated, and then determining the leader of that place. The question is solved by addressing the partial question first and then using its answer to further derive the answer to the whole question. So the type of this question is {composition}.
""",
    "conjunction": """Question: What educational institution that has a mascot named Mercy Mavericks did Mark Zuckerberg attend?
Answer: This question involves two independent sub-questions: which educational institution has a mascot called Mercy Mavericks, and which educational institutions did Mark Zuckerberg attend. The answer to the main question is obtained by taking the intersection of the answers to these two sub-questions. So the type of this question is {conjunction}.
""",
    "comparative": """Question: Who was the governor of Georgia in 2011, that their government position terminated before March 21, 2010?
Answer: This question involves comparing the termination time of the governor of Georgia in 2011 with March 21, 2010. So the type of this question is {comparative}.
""",
    "superlative": """Question: What home of Tennessee Williams had the largest location GNIS feature ID?
Answer: This question involves a superlative: The task is to find the location of Tennessee Williams' home with the largest GNIS feature ID. So the type of this question is {superlative}.
"""
}


prompt_decompose_group = {
    "composition": """Use multi-step reasoning and decompose the question into several sub-questions. If a subsequent subquestion requires the answer to the previous subquestion, replace the answer to subquestion-k with [#k] in the subsequent subquestion.

Examples:
Question: Which college was attended by the artist who headlined the John Mayer 2008 Summer Tour?
Answer: subquestion-1: {{Who is the artist that headlined the John Mayer 2008 Summer Tour?}}
subquestion-2: {{Which college was attended by [#1]?}}

Question: The people from Masjid Hamza, Valley Stream places of worship pray to whom?
Answer: subquestion-1: {{What religious group worships at Masjid Hamza, Valley Stream?}}
subquestion-2: {{To whom do [#1] pray?}}

Question: Find the public speaker who spoke on the Berlin Wall, what city was he later assasinated in?
Answer: subquestion-1: {{Who is the public speaker who spoke on the Berlin Wall?}}
subquestion-2: {{In what city was [#1] assassinated?}}

Question for you (Generate the subquestions strictly in the format of given examples, without adding anything else):
Question: {}
Answer:""",

    "conjunction": """Use multi-step reasoning and decompose the question into several sub-questions. If a subsequent subquestion requires the answer to the previous subquestion, replace the answer to subquestion-k with [#k] in the subsequent subquestion.

Examples:
Question: What sports team owned by The Madison Square Garden Company did Tyson Chandler play for?
Answer: subquestion-1: {{What sports team did Tyson Chandler play for?}}
subquestion-2: {{What sports team is owned by The Madison Square Garden Company?}}
subquestion-3: {{What is the intersection of set ([#1]) and set ([#2])?}}

Question: What geographic region through which the Appalachian Mountains extend is the location whose central government is in Boston?
Answer: subquestion-1: {{What is the geographic region through which the Appalachian Mountains extend?}}
subquestion-2: {{What is the location whose central government is in Boston?}}
subquestion-3: {{What is the intersection of set ([#1]) and set ([#2])?}}

Question: In which location, does Bradley Walsh live, where the location time zone, is in the Pacific Time Zone?
Answer: subquestion-1: {{In which location does Bradley Walsh live?}}
subquestion-2: {{Which location is in the Pacific Time Zone?}}
subquestion-3: {{What is the intersection of set ([#1]) and set ([#2])?}}

Question for you (Generate the subquestions strictly in the format of given examples, without adding anything else):
Question: {}
Answer:""",

    "comparative": """Use multi-step reasoning and decompose the question into several sub-questions. If a subsequent subquestion requires the answer to the previous subquestion, replace the answer to subquestion-k with [#k] in the subsequent subquestion.

Examples:
Question: Which museum that was founded before 1893-09-16 is a good place to visit in downtown Chicago?
Answer: subquestion-1: {{What are the museums located in downtown Chicago?}}
subquestion-2: {{Among the museums in ([#1]), which ones were founded before 1893-09-16?}}

Question: What is there to do in San Diego for fun that is a zoo that opened after 1915?
Answer: subquestion-1: {{What are the zoos located in San Diego?}}
subquestion-2: {{Among the zoos in ([#1]), which ones opened after 1915?}}

Question: Who did the Cleveland Cavaliers draft, which is the pro athlete who started his sports career on, or after the year of 2003?
Answer: subquestion-1: {{Which pro athlete did the Cleveland Cavaliers draft?}}
subquestion-2: {{Among ([#1]), who started his sports career on or after the year 2003?}}

Question for you (Generate the subquestions strictly in the format of given examples, without adding anything else):
Question: {}
Answer:""",

    "superlative": """Use multi-step reasoning and decompose the question into several sub-questions. If a subsequent subquestion requires the answer to the previous subquestion, replace the answer to subquestion-k with [#k] in the subsequent subquestion.

Examples:
Question: Which person that held a government position earliest did Baron de Montesquie influence?
Answer: subquestion-1: {{Who are the people influenced by Baron de Montesquie?}}
subquestion-2: {{Among the people in ([#1]), who held a government position earliest?}}

Question: What zoo that opened earliest is there to see in Dallas, TX?
Answer: subquestion-1: {{What are the zoos located in Dallas, TX?}}
subquestion-2: {{Among the zoos in ([#1]), which one opened earliest?}}

Question: What nation where the real adjusted value of the GDP was smallest is the birthplace of Nina Dobrev?
Answer: subquestion-1: {{What is the nationality of Nina Dobrev?}}
subquestion-2: {{Among ([#1]), which nation has the smallest real adjusted value of GDP?}}

Question for you (Generate the subquestions strictly in the format of given examples, without adding anything else):
Question: {}
Answer:"""
}


prompt_integrate_judge = """Answer the question \"{{answer entity}}\" according to the given Q&A references and your knowledge.
Question: Which languages are recognized as the official ones in the nation where Trelawny Parish is located?
References: Which nation is Trelawny Parish located in? -- Jamaica
What are the official languages of Jamaica? -- Jamaican English
Answer: Trelawny Parish is located in Jamaica, and the official language of Jamaica is Jamaican English. So the given references are [sufficient] for answering the question, and the answer is {{Jamaican English}}.

Question: What major trading partner of Germany has a capital called Kabul?
References: What is the major trading partner of Germany? -- United States of America
What country has a capital called Kabul? -- Afghanistan
What is the intersection of set (United States of America) and set (Afghanistan)? -- The intersection of set (United States of America) and set (Afghanistan) is empty, because the given knowledge triplets do not establish any relation or intersection between these two sets.
Answer: The given references are [insufficient] for answering the question. But based on my knowledge, Afghanistan is also a major trading partner of Germany, and its captical is Kabul. So the answer is {{Afghanistan}}.

Question: What sports team owned by The Madison Square Garden Company did Tyson Chandler play for?
References: What sports team did Tyson Chandler play for? -- New York Knicks, Dallas Mavericks
What sports team is owned by The Madison Square Garden Company? -- New York Knicks, New York Liberty, New York Rangers, Hartford Wolf Pack
What is the intersection of set (New York Knicks, Dallas Mavericks) and set (New York Knicks, New York Liberty, New York Rangers, Hartford Wolf Pack)? -- New York Knicks
Answer: Tyson Chandler played for New York Knicks, which is a sports team owned by The Madison Square Garden Company. So the given references are [sufficient] for answering the question, and the answer is {{New York Knicks}}.

Question: Who knows the art style created by the artist that produced the Hospital in Arles produce?
References: Who is the artist that produced the Hospital in Arles? -- Vincent van Gogh
What is the art style created by Vincent van Gogh? -- Post-Impressionism
Who knows (Post-Impressionism)? -- Vincent van Gogh
Answer: The artist that produced the Hospital in Arles is Vincent van Gogh, and the art style he created is Post-Impressionism. So the given references are [sufficient] for answering the question, and the answer is {{Post-Impressionism}}.

Question: Which Colorado representative held their governmental position since 2009-01-03?
References: Who are Colorado representatives? -- Charles J. Hughes, Jr.;Gordon L. Allott;Rice W. Means;Michael Bennet
Among (Charles J. Hughes, Jr.;Gordon L. Allott;Rice W. Means;Michael Bennet), which ones held their governmental position since 2009-01-03? -- unknown
Answer: The given references are [insufficient] for answering the question. But based on my knowledge, among (Charles J. Hughes, Jr.; Gordon L. Allott; Rice W. Means; Michael Bennet), Michael Bennet held his governmental position since 2009-01-21, which is later than 2009-01-03. So the answer is {{Michael Bennet}}.

Question: What countries in South American had the CO2 emissions per capita of 2009 metric ton?
References: What are the countries in South American? -- Empire of Brazil, Brazil, Peru, Colombia, Uruguay, Venezuela, Argentina, Bolivia
Among the countries in (Empire of Brazil, Brazil, Peru, Colombia, Uruguay, Venezuela, Argentina, Bolivia), which had the CO2 emissions per capita of 2009 metric ton? -- Brazil, Uruguay, Venezuela
Answer: Based on the references, among the South American countries, Brazil, Uruguay, and Venezuela had the CO2 emissions per capita of 2009 metric ton. So the given references are [sufficient] for answering the question, and the answer are {{Brazil}}, {{Uruguay}} and {{Venezuela}}.

Question: What should be done with kids in the location where the Phoenix New Times newspaper is circulated?
References: Where is the Phoenix New Times newspaper circulated? -- Phoenix
What are the tourist attractions in Phoenix? -- Phoenix Art Museum
What activities or events can be held in Phoenix Art Museum? -- Phoenix Art Museum offers art exhibitions, workshops, lectures, film screenings, family activities, live performances, and special events.
Answer: Based on the references, the Phoenix New Times newspaper is circulated in Phoenix, and kids can go to the Phoenix Art Museum. So the given references are [sufficient] for answering the question, and the answer is {{Phoenix Art Museum}}.

Question: What team that won 1995 AFC Championship and once had Jim Harbaugh on its roster?
References: What team won the 1995 AFC Championship? -- San Diego Chargers
What team once had Jim Harbaugh on its roster? -- Indianapolis Colts and San Diego Chargers
What is the intersection of set (San Diego Chargers) and set (Indianapolis Colts and San Diego Chargers)? -- San Diego Chargers
Answer: Based on the references, the team that satisfies both "won the 1995 AFC Championship" and "had Jim Harbaugh on its roster" is San Diego Chargers. So the given references are [sufficient] for answering the question, and the answer is {{San Diego Chargers}}.

Question: Where does the actor who played Beatrice Bell live currently?
References: Who is the actor that played Beatrice Bell? -- Avril Lavigne
Where does Avril Lavigne live? -- The answer is not provided by the given knowledge triplets. However, based on my knowledge, Avril Lavigne's current place of residence is unknown as he keeps his personal life quite private.
Answer: The given references are [insufficient] for answering the question, only that Avril Lavigne is the actor who played Beatrice Bell. But based on my knowledge, Avril Lavigne lives in Greater Napanee and Belleville. So the answer are {{Greater Napanee}} and {{Belleville}}.

Question: Which celebrities who dated Kim Khardashian have appeared on television after June 23, 2009?
References: Who dates Kim Khardashian? -- Nick Cannon
What was the television program that Nick Cannon appeared on? -- The Winner Is
After what date did The Winner Is appeared on television? -- after 2012
Answer: Based on the references, Nick Cannon dated Kim Kardashian and appeared on the television program "The Winner Is" after 2012. So it's confirmed that Nick Cannon appeared on television after June 23, 2009. So the given references are [sufficient] for answering the question, and the answer is {{Nick Cannon}}.

Question: {}
References: {}
Answer:"""


prompt_answer_io = """Question: who was the wife of king edward vii
Answer: {{Alexandra of Denmark}}

Question: Which languages are recognized as the official ones in the nation where Trelawny Parish is located?
Answer: {{Jamaican English}}

Question: what position did john stockton play
Answer: {{Point guard}}

Question: Which celebrities who dated Kim Khardashian have appeared on television after June 23, 2009?
Answer: {{Nick Cannon, Nick Lachey}}

Quesiton for you (Generate the answer strictly in the format of given examples, without adding anything else):
Question: {}
Answer:"""


prompt_answer_cot = """Question: who was the wife of king edward vii
Answer: King Edward VII was married to Alexandra of Denmark. So the answer is {{Alexandra of Denmark}}.

Question: Which languages are recognized as the official ones in the nation where Trelawny Parish is located?
Answer: Trelawny Parish is located in Jamaica, and the official language in Jamaica is Jamaican English. So the answer is {{Jamaican English}}.

Question: what position did john stockton play
Answer: John Stockton was an American professional basketball player, and he played as a Point guard. So the answer is {{Point guard}}.

Question: Which celebrities who dated Kim Khardashian have appeared on television after June 23, 2009?
Answer: The celebrities who dated Kim Khardashian are Reggie Bush, Ray J, Nick Cannon, Joe Francis, Nick Lachey, and Ben Roethlisberger. Among these people, Nick Cannon appeared on TV at January 1, 2012, Nick Lachey appeared on TV at June 25, 2013 and December 14, 2009. Both of them have appeared on TV after June 23, 2009. So the answer is {{Nick Cannon, Nick Lachey}}.

Question: {}
Answer:"""