import random


TASK_DICT = {
    'dbpedia-entity': 'Given a query, retrieve relevant entity descriptions from DBPedia.',
    'arguana': 'Given a claim, find documents that refute the claim.',
    'climate-fever': 'Given a claim about climate change, retrieve documents that support or refute the claim.',
    'cqadupstack': 'Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question.',
    'fever': 'Given a claim, retrieve documents that support or refute the claim.',
    'fiqa': 'Given a financial question, retrieve user replies that best answer the question.',
    'hotpotqa': 'Given a multi-hop question, retrieve documents that can help answer the question.',
    'msmarco': 'Given a web search query, retrieve relevant passages that answer the query.',
    'nfcorpus': 'Given a question, retrieve relevant documents that best answer the question.',
    'nq': 'Given a question, retrieve Wikipedia passages that answer the question.',
    'quora': 'Given a question, retrieve questions that are semantically equivalent to the given question.',
    'scidocs': 'Given a scientific paper title, retrieve paper abstracts that are cited by the given paper.',
    'scifact': 'Given a scientific claim, retrieve documents that support or refute the claim.',
    'webis-touche2020': 'Given a question, retrieve detailed and persuasive arguments that answer the question.',
    'trec-covid': 'Given a query on COVID-19, retrieve documents that answer the query.',
    'arxiv': 'Given a question, retrieve passages that answer the question.',
    'news': 'Given a question, retrieve passages that answer the question.',
    'finance': 'Given a question, retrieve passages that answer the question.',
    'healthcare': 'Given a question, retrieve passages that answer the question.',
    'law': 'Given a question, retrieve passages that answer the question.'
}


QUERY_TYPE_DICT = {
    'dbpedia-entity': 'query',
    'arguana': 'claim',
    'climate-fever': 'claim',
    'cqadupstack': 'question',
    'fever': 'claim',
    'fiqa': 'financial question',
    'hotpotqa': 'multi-hop question',
    'msmarco': 'web search query',
    'nfcorpus': 'question',
    'nq': 'question',
    'quora': 'question',
    'scidocs': 'scientific paper title',
    'scifact': 'scientific claim',
    'webis-touche2020': 'question',
    'trec-covid': 'query',
    'arxiv': 'question',
    'news': 'question',
    'finance': 'question',
    'healthcare': 'question',
    'law': 'question'
}


PASSAGE_TYPE_DICT = {
    'dbpedia-entity': 'entity description',
    'arguana': 'document',
    'climate-fever': 'document',
    'cqadupstack': 'question description',
    'fever': 'document',
    'fiqa': 'user reply',
    'hotpotqa': 'document',
    'msmarco': 'passage',
    'nfcorpus': 'document',
    'nq': 'Wikipedia passage',
    'quora': 'question',
    'scidocs': 'paper abstract',
    'scifact': 'document',
    'webis-touche2020': 'argument',
    'trec-covid': 'document',
    'arxiv': 'passage',
    'news': 'passage',
    'finance': 'passage',
    'healthcare': 'passage',
    'law': 'passage'
}


QG_MISSION_DICT = {
    'dbpedia-entity': 'Generate a {query_type} that is relevant to the {passage_type} from DBPedia.',
    'arguana': 'Generate a {query_type} that can be refuted by the {passage_type}.',
    'climate-fever': 'Generate a {query_type} about climate change that can be {argu_type} by the {passage_type}.',
    'cqadupstack': 'Generate a {query_type} that is a duplicate to the given question description from Stackexchange.',
    'fever': 'Generate a {query_type} that can be {argu_type} by the {passage_type}.',
    'fiqa': 'Generate a {query_type} that the {passage_type} can answer.',
    'hotpotqa': 'Generate a {query_type} that the {passage_type} can answer.',
    'msmarco': 'Generate a {query_type} that is relevant to the {passage_type}.',
    'nfcorpus': 'Generate a {query_type} that the {passage_type} can answer.',
    'nq': 'Generate a {query_type} that the {passage_type} can answer.',
    'quora': 'Generate a {query_type} that is semantically equivalent to the given {passage_type}.',
    'scidocs': 'Generate a {query_type} that suggests the corresponding paper cites the paper with the given {passage_type}.',
    'scifact': 'Generate a {query_type} that can be {argu_type} by the {passage_type}.',
    'webis-touche2020': 'Generate a Yes/No {query_type} that the detailed and persuasive {passage_type} can answer.',
    'trec-covid': 'Generate a {query_type} on COVID-19 that the {passage_type} can answer.',
    'arxiv': 'Generate a {query_type} that the {passage_type} can answer.',
    'news': 'Generate a {query_type} that the {passage_type} can answer.',
    'finance': 'Generate a {query_type} that the {passage_type} can answer.',
    'healthcare': 'Generate a {query_type} that the {passage_type} can answer.',
    'law': 'Generate a {query_type} that the {passage_type} can answer.'
}


QUERY_LENGTH_LIST = ['less than 5 words'] * 2 + \
    ['5 to 10 words'] * 5 + \
    ['10 to 15 words'] * 4 + \
    ['at least 15 words']


CLAIM_QUERY_LENGTH_LIST = ['10 to 15 words'] * 5 + \
    ['at least 15 words'] * 2


CLAIM_DATASET_LIST = [
    'climate-fever',
    'fever',
    'scifact',
]

EXAMPLE_DATASET_LIST = [
    'climate-fever',
    'webis-touche2020',
]


QUERY_STYLE_LIST = ['plain and simple'] * 4 + \
    ['common and formal'] * 4 + \
    ['casual and informal'] * 2 + \
    ['professional and complex']


EXAMPLES_DICT = {
    "climate-fever": [
        {
            "passage": "Habitat destruction\nHabitat destruction is the process in which natural habitat is rendered unable to support the species present. In this process, the organisms that previously used the site are displaced or destroyed, reducing biodiversity. Habitat destruction by human activity is mainly for the purpose of harvesting natural resources for industry production and urbanization. Clearing habitats for agriculture is the principal cause of habitat destruction. Other important causes of habitat destruction include mining, logging, trawling and urban sprawl. Habitat destruction is currently ranked as the primary cause of species extinction worldwide. It is a process of natural environmental change that may be caused by habitat fragmentation, geological processes, climate change or by human activities such as the introduction of invasive species, ecosystem nutrient depletion, and other human activities   The terms habitat loss and habitat reduction are also used in a wider sense, including loss of habitat from other factors, such as water and noise pollution.",
            "query": "Global warming is driving polar bears toward extinction.",
        },
        {
            "passage": "Coral bleaching\nCoral bleaching occurs when coral polyps expel algae that lives inside their tissues. Normally, coral polyps live in an endosymbiotic relationship with the algae and that relationship is crucial for the coral and hence for the health of the whole reef. Bleached corals continue to live. But as the algae provide the coral with 90%% of its energy, after expelling the algae the coral begins to starve. Above-average sea water temperatures caused by global warming have been identified as a leading cause for coral bleaching worldwide. Between 2014 and 2016, the longest global bleaching events ever were recorded. According to the United Nations Environment Programme, these bleaching events killed coral on an unprecedented scale. In 2016, bleaching hit 90 percent of coral on the Great Barrier Reef and killed 29 percent of the reef 's coral. In 2017, the bleaching further expanded to areas of the park that were previously spared, such as the central one.   __TOC__",
            "query": "The Great Barrier Reef is experiencing the most widespread bleaching ever recorded.",
        },
        {
            "passage": "Sea level rise\nA sea level rise is an increase in the volume of water in the world 's oceans, resulting in an increase in global mean sea level. Sea level rise is usually attributed to global climate change by thermal expansion of the water in the oceans and by melting of Ice sheets and glaciers on land. Melting of floating ice shelves or icebergs at sea raises sea levels only slightly.   Sea level rise at specific locations may be more or less than the global average. Local factors might include tectonic effects, subsidence of the land, tides, currents, storms, etc..  Sea level rise is expected to continue for centuries. Because of the slow inertia, long response time for parts of the climate system, it has been estimated that we are already committed to a sea-level rise of approximately 2.3 m for each degree Celsius of temperature rise within the next 2,000 years. IPCC Summary for Policymakers, AR5, 2014, indicated that the global mean sea level rise will continue during the 21st century, very likely at a faster rate than observed from 1971 to 2010. Projected rates and amounts vary. A January 2017 NOAA report suggests a range of GMSL rise of 0.3--2.5m possible during the 21st century.   Sea level rises can considerably influence human populations in coastal and island regions and natural environments like marine ecosystems.",
            "query": "Sea level rise has been slow and a constant, pre-dating industrialization.",
        }
    ],
    "webis-touche2020": [
        {
            "passage": "Bloomberg's Ban on E-Cigs\nElectronic cigarettes comes with different cartridges including 6-18mg of nicotine and sometimes 0mg. This is to say that electronic cigarettes are safer to smoke than traditional cigarettes. Electronic cigarettes do not cause tar because of the fact that it does not contain tobacco and leave behind no tar. As a result, the main components of carcinogen are not present to create a problem that traditional cigarettes that contain various chemicals, additives and smokes. Vapor is just vapor. It does not include any smell or lingering odor. It is far from affecting people around you while smoking electronic cigarette. Electronic cigarettes should not be banned because it does not pose any harm to its users and help people from quitting cigar.",
            "query": "Is vaping with e-cigarettes safe?",
        },
        {
            "passage": "corporal punishment\nShould corporal punishment be be banned or kept in schools, daycares, etc? I am a student and I think that with the way children/teens act in today's society they need to be disciplined in some way shape or form. Give me your opinions, should we bring it back or not? If we have more punishment in schools and daycares just think how much more respect kids would give their parents. I think it should be brought back, and kept.",
            "query": "Should corporal punishment be used in schools?",
        },
        {
            "passage": "Nuclear energy is a crucial alternative energy source that is too valuable to be restricted.\nWhile none can truly replace fossil fuels, only one source is currently a contributor strong enough to supply a large portion of what fossil fuels power now, and that's nuclear energy. Nuclear energy may well be the only possible candidate that produces anything nearly as close to what fossil fuel sources do now while being committed to significantly reducing carbon emissions. Currently the third largest source, nuclear energy supplies about a sixth of all electricity generation in the world, only slightly less than hydro power. Nuclear power plants are far more gross-land efficient than both fossil-fuel plants and hydro-electric plants and have much potential to expand throughout the world. Moreover, experts predict that nuclear energy will be a sustainable source for 30,000-60,000 years. It is also expected that energy security will be considerably reliable considering the widely available 16million metric tons of uranium. While being the only feasible large-scale alternative to fossil-fuels, nuclear energy is also an excellent method in curbing carbon emissions. In the US, nuclear energy provided about a fifth of all produced electricity, saving 700 million metric tons of CO2 emissions yearly, an amount that matches the amount from all US passenger car exhaust. As a source with such potential, limiting expansion is simply putting a choke-hold on our future.",
            "query": "Can alternative energy effectively replace fossil fuels?",
        }
    ],
    "arguana": [
        {
            "passage": "global law international law politics defence warpeace house supports new New START will cause American missile and nuclear capabilities to atrophy, not to be maintained. This is because it locks the US in to agreements of defensive reductions which are tied into Russian offensive reductions. This could eventually leave the US badly under-defended by its missile systems when compared against the offensive capabilities of other nuclear states. Moreover, New START leaves in place the pre-existing Russian tactical nuclear advantage harming US capabilities by comparison. [1] Overall New START hams US missile and nuclear capabilities, and further advantages Russia and other nuclear powers, and so should not be supported. As Mitt Romney argued in 2010: \"Does New START limit America\u2019s options for missile defense? Yes. For the first time, we would agree to an interrelationship between strategic offensive weapons and missile defense. Moreover, Russia already asserts that the document would constitute a binding limit on our missile defense program. But the WikiLeaks revelation last weekend that North Korea has supplied Iran with long-range Russian missiles confirms that robust missile defense is urgent and indispensable.\" [2]  [1] Spring, Baker. \"Twelve Flaws of New START That Will Be Difficult to Fix\". Heritage Foundation, The Foundry. 16 September 2010.   [2] Romney, Mitt. \"Stop START.\" Boston.com. 3 December 2010.",
            "query": "The New START treaty maintains US nuclear and missile defence.  The US\u2019 Nuclear armament will be modernized along with New START. \u201cThe Obama administration has agreed to provide for modernization of the infrastructure essential to maintaining our nuclear arsenal. Funding these efforts has become part of the negotiations in the ratification process. The administration has put forth a 10-year plan to spend $84 billion on the Energy Department's nuclear weapons complex. Much of the credit for getting the administration to add $14 billion to the originally proposed $70 billion for modernization goes to Sen. Jon Kyl, the Arizona Republican who has been vigilant in this effort. Implementing this modernization program in a timely fashion would be important in ensuring that our nuclear arsenal is maintained appropriately over the next decade and beyond.\u201d [1]  Both US Military and civilian leaders insist that the new START treaty will still allow the US to deploy effective missile defenses, something which Russia was opposed to, and so will not affect US missile defense plans. The main limit on missile defense is that the treaty prevents the conversion of existing launchers for this purpose this would be more expensive than building new missiles specifically for defense purposes. [2]  Furthermore, as Joe Biden argues, New START is important to Russian cooperation on missile defense: \"This [missile defense] system demonstrates America's enduring commitment to Article 5 of the Washington Treaty\u2014that an attack on one is an attack on all. NATO missile defense also provides the opportunity for further improvements in both NATO-Russian and U.S.-Russian relations. NATO and Russia agreed at Lisbon to carry out a joint ballistic missile threat assessment, to resume theater missile-defense exercises, and to explore further cooperation on territorial missile defense\u2014things that were nearly unimaginable two years ago. These agreements underscore the strategic importance the alliance attaches to improving its relationship with Russia. But trust and confidence in our relationship with Russia would be undermined without Senate approval of the New Start Treaty, which reduces strategic nuclear forces to levels not seen since the 1950s, and restores important verification mechanisms that ceased when the first Start Treaty expired last December.\" [3]  In many ways, in the 21st Century having an abundance of nuclear weapons, particularly having too many, is more of a liability than an advantage. The United States will be far safer with fewer nuclear weapons in the world and a stronger, more stable relationship with Russia under New START, and this is desirable. Therefore it is clear that New START maintains the important parts of US nuclear capabilities while removing the over-abundance which may become a liability due to security and medical concerns, and so New START should be supported.  [1] Kissinger, Henry A. ; Shultz, George P. ; Baker III, James A\u2019 ; Eagleburger , Lawrence S. ; and Powell, Colin L. \"The Republican case for ratifying New START\". Washington Post. 2 December 2010.   [2] ibid  [3] Biden, Joseph. \"The case for ratifying New START\". Wall Street Journal. 25 November 2010."
        },
        {
            "passage": "defence house believes all nations have right nuclear weapons The threat represented by potential nuclear powers will instigate pre-emptive strikes by countries fearing the future behaviour of the budding nuclear powers. Until a state develops a nuclear capacity that its rivals believe they cannot destroy in a first strike, nuclear weapons increase the risk of war. For example, Israel will have a very real incentive to attack Iran before it can complete its development of nuclear weapons, lest it become an existential threat to Israel\u2019s survival. The United States military even considered attempting to destroy the USSR\u2019s capability before they had second strike capability General Orvil Anderson publicly declared: \u201cGive me the order to do it and I can break up Russia\u2019s five A-bomb nests in a week\u2026And when I went up to Christ\u2014I think I could explain to Him that I had saved civilization.\u201d [1] The development of nuclear weapons can thus destabilize regions before they are ever operational, as it is in no country\u2019s interest that its rivals become capable of using nuclear force against it. Clearly, it is best that such states do not develop nuclear weapons in the first place so as to prevent such instability and conflict.  [1] Stevens, Austin \u201cGeneral Removed over War Speech,\u201d New York Times, September 2, 1950, p. 8  improve this  If a country is surrounded by hostile neighbours that are likely to attempt a pre-emptive strike upon it, then nuclear weapons are all the more desirable. With nuclear weapons a country cannot be pushed around by regional bullies. It seems perfectly fair that Iran would covet the ability to resist Israeli might in the Middle East and defend itself from aggression by it or the United States.",
            "query": "The threat of a state developing nuclear weapons could instigate pre-emptive strikes from its neighbours and rivals to prevent the acquisition of such weapons  The threat represented by potential nuclear powers will instigate pre-emptive strikes by countries fearing the future behaviour of the budding nuclear powers. Until a state develops a nuclear capacity that its rivals believe they cannot destroy in a first strike, nuclear weapons increase the risk of war. For example, Israel will have a very real incentive to attack Iran before it can complete its development of nuclear weapons, lest it become an existential threat to Israel\u2019s survival. The United States military even considered attempting to destroy the USSR\u2019s capability before they had second strike capability General Orvil Anderson publicly declared: \u201cGive me the order to do it and I can break up Russia\u2019s five A-bomb nests in a week\u2026And when I went up to Christ\u2014I think I could explain to Him that I had saved civilization.\u201d [1] The development of nuclear weapons can thus destabilize regions before they are ever operational, as it is in no country\u2019s interest that its rivals become capable of using nuclear force against it. Clearly, it is best that such states do not develop nuclear weapons in the first place so as to prevent such instability and conflict.  [1] Stevens, Austin \u201cGeneral Removed over War Speech,\u201d New York Times, September 2, 1950, p. 8  improve this  COUNTERPOINT  If a country is surrounded by hostile neighbours that are likely to attempt a pre-emptive strike upon it, then nuclear weapons are all the more desirable. With nuclear weapons a country cannot be pushed around by regional bullies. It seems perfectly fair that Iran would covet the ability to resist Israeli might in the Middle East and defend itself from aggression by it or the United States."
        },
        {
            "passage": "imate international global house believes outcome paris climate conference The United States Senate would be a potential sticking point for any treaty however it would be unlikely that the United States would hold out against the rest of the world. At the worst case it would simply sign next time the democrats gain a majority.",
            "query": "A more informal agreement avoids the US congress  The United States Congress is a potential hurdle for any climate agreement. While President Barack Obama is keen to make tackling climate change a legacy of his Presidency the Republican dominated Congress is both likely to try to block the President for that very reason and is sceptical of climate change. It is therefore a major benefit to have an agreement that will not need to be submitted to Congress for approval as any treaty needs to be confirmed by the Senate.  The Secretary of State Kerry argues that it is \u201cdefinitely not going to be a treaty,\u201d and \u201cnot going to be legally binding reduction targets like Kyoto\u201d. It won\u2019t need to be passed to the Senate because the President already has the power to implement the agreement through existing law. [1]  [1] Mufson, Steven, and Demirjian, Karoun, \u2018Trick or treaty? The legal question hanging over the Paris climate change conference\u2019, Washington Post, 30 November 2015,"
        }
    ],
    "trec-covid": [
        {
            "passage": "Evaluation of the clinical characteristics of suspected or confirmed cases of COVID-19 during home care with isolation: A new retrospective analysis based on O2O Summary Background The recent outbreak of the novel coronavirus in December 2019 (COVID-19) has activated top-level response nationwide. We developed a new treatment model based on the online-to-offline (O2O) model for the home isolated patients, because in the early stages the medical staff were insufficient to cope with so many patients. Methods In this single-centered, retrospective study, we enrolled 48 confirmed/suspected COVID-19 patients who underwent home isolation in Wuhan between January 6 and January 31, 2020. By WeChat and online document editing all patients were treated with medical observation scale. The clinical indications such as Fever, Muscle soreness, Dyspnea and Lack of strength were collected with this system led by medical staff in management, medicine, nursing, rehabilitation and psychology. Findings The mean(SD) age of 48 patients was 39.08(13.88) years, 35(72.9%) were women. Compared with non-hospitalized patients, inpatients were older(\u22658805;70years, 2.4% vs 33.3%, P<0.04). All inpatients had fever, 50% inpatients had coughs and showed infiltration in both lungs at the time of diagnosis. 33.3% inpatients exhibited negative changes in their CT results at initial diagnosis. The body temperature of non-hospitalized patients with mild symptoms returned to normal by day 4-5. While dyspnea peaked on day 6 for non-hospitalized patients with mild symptoms, it persisted in hospitalized patients and exacerbated over time. The lack of strength and muscle soreness were both back to normal by day 4 for non-hospitalized patients. Interpretation Monitoring the trends of symptoms is more important for identifying severe cases. Excessive laboratory data and physical examination are not necessary for the evaluation of patients with mild symptoms. The system we developed is the first to convert the subjective symptoms of patients into objective scores. This type of O2O, subjective-to-objective strategy may be used in regions with similar highly infectious diseases to minimize the possibility of infection among medical staff.",
            "query": "what are best practices in hospitals and at home in maintaining quarantine?"
        },
        {
            "passage": "Increased Detection coupled with Social Distancing and Health Capacity Planning Reduce the Burden of COVID-19 Cases and Fatalities: A Proof of Concept Study using a Stochastic Computational Simulation Model Objective: In absence of any vaccine, the Corona Virus Disease 2019 (COVID-19) pandemic is being contained through a non-pharmaceutical measure termed Social Distancing (SD). However, whether SD alone is enough to flatten the epidemic curve is debatable. Using a Stochastic Computational Simulation Model, we investigated the impact of increasing SD, hospital beds and COVID-19 detection rates in preventing COVID-19 cases and fatalities. Research Design and Methods: The Stochastic Simulation Model was built using the EpiModel package in R. As a proof of concept study, we ran the simulation on Kasaragod, the most affected district in Kerala. We added 3 compartments to the SEIR model to obtain a SEIQHRF (Susceptible-Exposed-Infectious-Quarantined-Hospitalised-Recovered-Fatal) model. Results: Implementing SD only delayed the appearance of peak prevalence of COVID-19 cases. Doubling of hospital beds could not reduce the fatal cases probably due to its overwhelming number compared to the hospital beds. Increasing detection rates could significantly flatten the curve and reduce the peak prevalence of cases (increasing detection rate by 5 times could reduce case number to half). Conclusions: An effective strategy to contain the epidemic spread of COVID-19 in India is to increase detection rates in combination with SD measures and increase in hospital beds.",
            "query": "has social distancing had an impact on slowing the spread of COVID-19?"
        },
        {
            "passage": "Comprehensive overview of COVID-19 based on current evidence In December 2019, twenty-seven pneumonia patients with unknown causes originated in South China seafood market in Wuhan. The virus infection spread rapidly and swept through China in less than a month. Subsequently, the virus was proven a novel coronavirus and named SARS-CoV-2. The outbreak of novel coronavirus has been determined as a Public Health Emergency of International Concern (PHEIC) by WHO on January 31, 2020. Similar to other coronaviruses like the Middle East Respiratory Syndrome (MERS) CoV and Severe Acute Respiratory Syndrome (SARS) CoV, the novel coronavirus was reported to spread via respiratory droplets and close contact from human to human, which means the virus is highly infectious and dangerous. Unfortunately, till now the virus has spread to over 200 countries/territories/areas around the world and the Coronavirus Disease 2019 (COVID-19) outbreak is continuing to grow. Currently, information sharing and transparency are essential for risk assessment and epidemic control in all endemic areas. In this article, we compared SARS-CoV-2 with SARS-CoV and influenza virus, discussed current researching progress of COVID-19, including clinical characteristics, pathological changes, treatment measures, and so on.",
            "query": "How does the coronavirus differ from seasonal flu?"
        }
    ],
    "fiqa": [
        {
            "passage": "\"To keep it simple, let's say that A shares trade at 500 on average between April 2nd 2014 and April 1st 2015 (one year anniversary), then if C shares trade on average: The payment will be made either in cash or in shares within 90 days. The difficulties come from the fact that the formula is based on an average price over a year, which is not directly tradable, and that the spread is only covered between 1% and 5%. In practice, it is unlikely that the market will attribute a large premium to voting shares considering that Page&Brin keep the majority and any discount of Cs vs As above 2-3% (to include cost of trading + borrowing) will probably trigger some arbitrage which will prevent it to extend too much. But there is no guarantee. FYI here is what the spread has looked like since April 3rd:  * details in the section called \"\"Class C Settlement Agreement\"\" in the S-3 filing\"",
            "query": "Does it make sense to trade my GOOGL shares for GOOG and pocket the difference?"
        },
        {
            "passage": "There is no one answer to this question, but there are some generalities. Most exchanges make a distinction between the passive and the aggressive sides of a trade.  The passive participant is the order that was resting on the market at the time of the trade.  It is an order that based on its price was not executable at the time, and therefore goes into the order book.  For example, I'm willing to sell 100 shares of a stock at $9.98 but nobody wants to buy that right now, so it remains as an open order on the exchange. Then somebody comes along and is willing to meet my price (I am glossing over lots of details here).  So they aggressively take out my order by either posting a market-buy, or specifically that they want to buy 100 shares at either $9.98, or at some higher price. Most exchanges will actually give me, as the passive (i.e. liquidity making) investor a small rebate, while the other person is charged a few fractions of a cent.  Google found NYSEArca details, and most other exchanges make their fees public as well.  As of this writing the generic price charged/credited: But they provide volume discounts, and many of the larger deals do fall into another tier of volume, which provides a different price structure.",
            "query": "How much do brokerages pay exchanges per trade?"
        },
        {
            "passage": "The $50k is subject to the appropriate income taxes, which may include FICA taxes including the employer share if you are self employed. The after tax money can then be invested with the amount invested being the cost basis (I.e., if you invest $40k you will have a cost basis of $40k).  In future years you will have taxes due if any of those investments pay dividends (or capital gain distributions). Once you sell you will have a capital gain or loss that you will pay taxes on (or take a deduction if a loss). Now you can improve this picture if you are able to put some of your money into a retirement account (either a tax deductible or a ROTH). With retirement accounts you do not pay tax on the capital gains or dividends. If you use a tax deferred account your tax is higher but that is because you were also investing Uncle Sam's portion of your pay check.",
            "query": "Income Tax and Investments"
        }
    ],
    "scidocs": [
        {
            "passage": "Two Bitcoins at the Price of One? Double-Spending Attacks on Fast Payments in Bitcoin Bitcoin is a decentralized payment system that is based on Proof-of-Work. Bitcoin is currently gaining popularity as a digital currency; several businesses are starting to accept Bitcoin transactions. An example case of the growing use of Bitcoin was recently reported in the media; here, Bitcoins were used as a form of fast payment in a local fast-food restaurant. In this paper, we analyze the security of using Bitcoin for fast payments, where the time between the exchange of currency and goods is short (i.e., in the order of few seconds). We focus on doublespending attacks on fast payments and demonstrate that these attacks can be mounted at low cost on currently deployed versions of Bitcoin. We further show that the measures recommended by Bitcoin developers for the use of Bitcoin in fast transactions are not always effective in resisting double-spending; we show that if those recommendations are integrated in future Bitcoin implementations, double-spending attacks on Bitcoin will still be possible. Finally, we leverage on our findings and propose a lightweight countermeasure that enables the detection of doublespending attacks in fast transactions.",
            "query": "Trends, Tips, Tolls: A Longitudinal Study of Bitcoin Transaction Fees"
        },
        {
            "passage": "Adaptive Subgradient Methods for Online Learning and Stochastic Optimization We present a new family of subgradient methods that dynamica lly incorporate knowledge of the geometry of the data observed in earlier iterations to perfo rm more informative gradient-based learning. Metaphorically, the adaptation allows us to find n eedles in haystacks in the form of very predictive but rarely seen features. Our paradigm stems fro m recent advances in stochastic optimization and online learning which employ proximal funct ions to control the gradient steps of the algorithm. We describe and analyze an apparatus for adap tively modifying the proximal function, which significantly simplifies setting a learning rate nd results in regret guarantees that are provably as good as the best proximal function that can be cho sen in hindsight. We give several efficient algorithms for empirical risk minimization probl ems with common and important regularization functions and domain constraints. We experimen tally study our theoretical analysis and show that adaptive subgradient methods outperform state-o f-the-art, yet non-adaptive, subgradient algorithms.",
            "query": "Online Learning for Neural Machine Translation Post-editing"
        },
        {
            "passage": "A circularly polarized planar antenna modified for passive UHF RFID The majority of RFID tags are linearly polarized dipole antennas, but a few use a planar, dual-dipole antenna that facilitates circular polarization, but requires a three-terminal IC. In this paper, we present a novel way to achieve circular polarization with a planar antenna using a two-terminal IC. We present an intuitive methodology for design, and perform experiments that validate circular polarization. The results show that the tag exhibits strong circular polarization, but the precise axial ratio of the tag remains uncertain due to lack of precision in the experimental system.",
            "query": "Coupling-Feed Circularly Polarized RFID Tag Antenna Mountable on Metallic Surface"
        }
    ],
    "scifact": [
        {
            "passage": "The Rho GEFs LARG and GEF-H1 regulate the mechanical response to force on integrins How individual cells respond to mechanical forces is of considerable interest to biologists as force affects many aspects of cell behaviour. The application of force on integrins triggers cytoskeletal rearrangements and growth of the associated adhesion complex, resulting in increased cellular stiffness, also known as reinforcement. Although RhoA has been shown to play a role during reinforcement, the molecular mechanisms that regulate its activity are unknown. By combining biochemical and biophysical approaches, we identified two guanine nucleotide exchange factors (GEFs), LARG and GEF-H1, as key molecules that regulate the cellular adaptation to force. We show that stimulation of integrins with tensional force triggers activation of these two GEFs and their recruitment to adhesion complexes. Surprisingly, activation of LARG and GEF-H1 involves distinct signalling pathways. Our results reveal that LARG is activated by the Src family tyrosine kinase Fyn, whereas GEF-H1 catalytic activity is enhanced by ERK downstream of a signalling cascade that includes FAK and Ras.",
            "query": "Leukemia associated Rho guanine nucleotide-exchange factor represses RhoA in response to SRC activation."
        },
        {
            "passage": "Perinatal mortality in rural China: retrospective cohort study. OBJECTIVES To explore the use of local civil registration data to assess the perinatal mortality in a typical rural county in a less developed province in China, 1999-2000. DESIGN Retrospective cohort study. Pregnancies in a cohort of women followed from registration of pregnancy to outcome of infant seven days after birth. SETTING Routine family planning records in 20 rural townships in eastern China. SUBJECTS 3697 pregnancies registered by the local family planning system during 1999. MAIN OUTCOME MEASURES Abortions, stillbirths, early neonatal mortality, perinatal mortality. RESULTS Only three cases were lost to follow up. The average age of the women at pregnancy was 25.9 years. Three hundred and twelve pregnancies were aborted and 240 ended in miscarriage (total 552, 15%). The perinatal mortality rate was 69 per 1000 births, the rate of stillbirth was 24 per 1000 births, and the early neonatal mortality was 46 per 1000 live births. The early neonatal mortality was 29 in boys and 69 in girls per 1000 live births. The perinatal mortality rate increased notably with parity and was higher in townships having lower income per capita. CONCLUSIONS The family planning system at the most local level is a useful data source for studying perinatal mortality in rural China. The perinatal mortality rate in the study county was higher than previously reported for both rural and urban areas in China. The results by parity and sex of the infant raise concern over the impact of the one child policy.",
            "query": "The one-child policy has been successful in lowering population growth."
        },
        {
            "passage": "Reverse engineering of TLX oncogenic transcriptional networks identifies RUNX1 as tumor suppressor in T-ALL The TLX1 and TLX3 transcription factor oncogenes have a key role in the pathogenesis of T cell acute lymphoblastic leukemia (T-ALL). Here we used reverse engineering of global transcriptional networks to decipher the oncogenic regulatory circuit controlled by TLX1 and TLX3. This systems biology analysis defined T cell leukemia homeobox 1 (TLX1) and TLX3 as master regulators of an oncogenic transcriptional circuit governing T-ALL. Notably, a network structure analysis of this hierarchical network identified RUNX1 as a key mediator of the T-ALL induced by TLX1 and TLX3 and predicted a tumor-suppressor role for RUNX1 in T cell transformation. Consistent with these results, we identified recurrent somatic loss-of-function mutations in RUNX1 in human T-ALL. Overall, these results place TLX1 and TLX3 at the top of an oncogenic transcriptional network controlling leukemia development, show the power of network analyses to identify key elements in the regulatory circuits governing human cancer and identify RUNX1 as a tumor-suppressor gene in T-ALL.",
            "query": "Normal expression of RUNX1 has tumor-promoting effects."
        }
    ],
    "nfcorpus": [
        {
            "passage": "Kiwifruit improves bowel function in patients with irritable bowel syndrome with constipation. Irritable bowel syndrome (IBS) is a common functional disorder of the gastrointestinal system, and is characterized by abdominal pain, diarrhea (IBS/D), constipation (IBS/C), and alternating diarrhea and constipation (IBSC/A). The purpose of this study was to examine the impact of a four week kiwifruit intervention on bowel function in patients diagnosed with IBS/C. Fifty-four patients with IBS/C and 16 healthy adults participated in this study. All subjects participated in the 6 week, three phase study, which included a baseline phase (1 week), a dietary intervention period (4 weeks), and a post-intervention phase (1 week). Forty-one IBS/C patients and all healthy adults consumed two Hayward green (Actinida deliciosa var) kiwifruits per day for 4 weeks. Thirteen IBS/C patients in the control group took two placebo capsules per day for 4 weeks. Colon transit time was measured immediately prior to and following the intervention period. All subjects completed daily defecation records. After the 4-week intervention, weekly defecation frequency significantly increased in the IBS/C group of participants who consumed kiwifruit (p<0.05). Colon transit time significantly decreased (p=0.026) in the IBS/C group that consumed kiwi fruit. These findings suggest that kiwifruit consumption for 4 weeks shortens colon transit time, increases defecation frequency, and improves bowel function in adults diagnosed with IBS/C.",
            "query": "Kiwifruit for Irritable Bowel Syndrome"
        },
        {
            "passage": "Brain cancer associated with environmental lead exposure: evidence from implementation of a National Petrol-Lead Phase-Out Program (PLPOP) in Taiwa... BACKGROUND AND OBJECTIVE: In 1981, a Petrol-Lead Phase-Out Program (PLPOP) was launched in Taiwan for the abatement of environmental lead emissions. The present study was intended to examine whether the high Petrol-Lead Emission Areas (PLEA) would result in an increase in the incidence rate of brain cancer based on a national data bank. METHODS: The national brain cancer incidence data was obtained from the Taiwan National Cancer Registry. Age standardized incidence rates were calculated based on the 2000 WHO world standard population, and gasoline consumption data was obtained from the Bureau of Energy. The differences in the trend tests for age-standardized incidence rates of brain cancer between high, median, low, and small PLEA were analyzed. RESULTS: A significant increase was found from small to high PLEA in age-standardized incidence rates of brain cancer. By taking six possible confounders into account, the age-standardized incidence rates for brain cancer were highly correlated with the median and high PLEA by reference to the small PLEA. CONCLUSION: After being adjusted for a number of relevant confounders, it could be concluded that high PLEA might result in an increase in the incidence rate of brain cancer resulting from high lead exposures. Copyright \u00a9 2011 Elsevier Ltd. All rights reserved.",
            "query": "Artificial Food Colors and ADHD"
        },
        {
            "passage": "Moderate alcohol consumption during adult life, drinking patterns, and breast cancer risk Context Multiple studies have linked alcohol consumption to breast cancer risk, but the risk of lower levels of consumption has not been well quantified. In addition, the role of drinking patterns (i.e. frequency of drinking and \u201cbinge\u201d drinking) and consumption at different times of adult life are not well understood. Objective To evaluate the association of breast cancer with alcohol consumption during adult life, including quantity, frequency, and age at consumption. Design, Setting, and Participants Prospective observational study of 105,986 women enrolled in the Nurses\u2019 Health Study followed from 1980 until 2008 with early adult and eight updated alcohol assessments during this time. Main Outcome Measures Relative risks of developing invasive breast cancer. Results 7690 cases developed during 2.4 million person-years of follow-up. Increasing alcohol consumption was associated with increased breast cancer risk that was statistically significant at levels as low as 5.0-9.9 gm/day, equivalent to 3-6 drinks/week (RR 1.15 (95% CI 1.06-1.24) 332 cases/100,000 person-years). After controlling for cumulative alcohol intake, binge drinking, but not frequency of drinking, was associated with breast cancer risk. Alcohol intake both earlier and later in adult life was independently associated with risk. Conclusion Low levels of alcohol consumption were associated with a small increase in breast cancer risk, with the most consistent measure being cumulative alcohol intake throughout adult life. Alcohol intake both earlier and later in adult life was independently associated with risk.",
            "query": "Breast Cancer & Alcohol: How Much is Safe?"
        }
    ],
    "fever": [
        {
            "passage": "House of Balloons House of Balloons is the debut mixtape by Canadian singer The Weeknd . It was released as a free download on March 21 , 2011 , by XO . The mixtape was also released on his official website . Its music incorporates electronic and urban genres , including R&B and soul , along with trip hop , indie rock and dream pop tones . The contributions to the mixtape 's production came from Canadian record producers such as Doc McKinney , Zodiac and Illangelo , among others .   In September 2013 , The Weeknd revealed that the House of Balloons is a real place , located at 65 Spencer Ave in Toronto .",
            "query": "House of Balloons is by Queen."
        },
        {
            "passage": "Battle of the Trebia The Battle of the Trebia ( or Trebbia ) was the first major battle of the Second Punic War , fought between the Carthaginian forces of Hannibal and the Roman Republic in December of 218 BC , on or around the winter solstice . It was a resounding Roman defeat with heavy losses , and yet some 10,000 and more Romans , over 2.5 legions , survived on the field and retreated in order to Placentia ( Piacenza ) . In this battle , Hannibal got the better of the Romans by exercising the careful and innovative planning for which he was famous . The impetuous and short-sighted opposing general , the consul Tiberius Sempronius Longus , allowed himself to be provoked into a frontal assault under physically difficult circumstances and failed to see that he was being led into a trap .   The battle took place in the flat country of the Province of Piacenza on the left bank of the Trebbia River , a shallow , braided stream , not far south from its confluence ( from the south ) with the Po river . The battle is named for the river . Although the precise location is not known for certain , it is generally accepted as being visible from the Via Emilia , now paralleled by highway A21/E70 and a railroad trunk line , all of which come from Piacenza , a contemporaneously placed Roman colony ( though perhaps on an existing settlement ) , and cross the river north of where the Romans did in the battle . The area is possibly in the comune of Rottofreno at its main settlement , San Nicol\u00f2 a Trebbia , in the vicinity of the coordinates given at the head of this article.Over the course of more than two millennia the precise configuration of the Trebbia and its streams as well as that of the Po have changed geologically . Although the location of Placentia is believed to be roughly the same , the original surfaces are under new layers of sediment and the locations of the bends , the depths and widths of the streams all have changed . Construction in the area also has been extensive , not to mention the turning over of the soil and obliteration of features by heavy bombing when the bridges and rail lines were destroyed in World War II . The Trebbia and the Po are currently heavily diked .",
            "query": "The Battle of the Trebia was fought in Kyoto."
        },
        {
            "passage": "Python (programming language) Python is a widely used high-level programming language for general-purpose programming , created by Guido van Rossum and first released in 1991 . An interpreted language , Python has a design philosophy which emphasizes code readability ( notably using whitespace indentation to delimit code blocks rather than curly brackets or keywords ) , and a syntax which allows programmers to express concepts in fewer lines of code than possible in languages such as C++ or Java . The language provides constructs intended to enable writing clear programs on both a small and large scale .   Python features a dynamic type system and automatic memory management and supports multiple programming paradigms , including object-oriented , imperative , functional programming , and procedural styles . It has a large and comprehensive standard library .   Python interpreters are available for many operating systems , allowing Python code to run on a wide variety of systems . CPython , the reference implementation of Python , is open source software and has a community-based development model , as do nearly all of its variant implementations . CPython is managed by the non-profit Python Software Foundation .",
            "query": "Python lacks support for object-oriented programming."
        }
    ],
    "quora": [
        {
            "passage": "What are some good data structure projects?",
            "query": "What are some good projects in data structure?"
        },
        {
            "passage": "How can I forget a person whom I loved for 6 years?",
            "query": "How do I forget a person whom I loved?"
        },
        {
            "passage": "What if Iran won the Iran-Iraq war?",
            "query": "What if Iran would have won the Iran-Iraq War?"
        }
    ],
    "hotpotqa": [
        {
            "passage": "National symbols of Albania The National symbols of Albania are the symbols that are used in Albania to represent what is unique about the nation, reflecting different aspects of its culture and history. They may also be used in the Republic of Kosovo, Macedonia, Montenegro, Greece (Chameria), and Serbia (Pre\u0161evo Valley), and by the Arb\u00ebresh\u00eb in Italy.",
            "query": "The National symbols of Albania are used in this part of Serbia that are composed of which municipalities?"
        },
        {
            "passage": "Kate Horsley Kate Horsley (born 1952) is the pen name of Kate Parker, an author of numerous works of historical fiction three of which are rooted in the Old West. Parker is a professor of English at Central New Mexico Community College in Albuquerque. She has a lifelong flirtation with Zen after reading Alan Watts. Her published novels include:",
            "query": "When did the British philosopher whose works inspired Kate Horsley's interest in Zen die?"
        },
        {
            "passage": "Robert Smith (Illinois politician) Robert Smith (June 12, 1802 \u2013 December 21, 1867) was a U.S. Representative from Illinois, nephew of Jeremiah Smith and Samuel Smith of New Hampshire. Smith founded General Mills in 1856.",
            "query": "Robert Smith founded the multinational company headquartered in what city?"
        }
    ],
    "msmarco": [
        {
            "passage": "Famciclovir (Famvir) is sometimes used to treat the herpes virus that causes cold sores and genital herpes (as well as the virus that causes shingles). This medicine is available only by prescription and is taken orally in tablet form.",
            "query": "what is famvir prescribed for"
        },
        {
            "passage": "CPAP is a treatment that uses mild air pressure to keep your breathing airways open. It involves using a CPAP machine that includes a mask or other device that fits over your nose or your nose and mouth, straps to position the mask, a tube that connects the mask to the machine\u00e2\u0080\u0099s motor, and a motor that blows air into the tube.",
            "query": "medicare's definition of mechanical ventilation"
        }
    ],
    "nq": [
        {
            "passage": "Sentence clause structure A simple sentence consists of only one clause. A compound sentence consists of two or more independent clauses. A complex sentence has at least one independent clause plus at least one dependent clause.[1] A set of words with no independent clause may be an incomplete sentence, also called a sentence fragment.",
            "query": "what kind of sentence contains an independent clause and a dependent clause"
        },
        {
            "passage": "Peter Beardsley Peter Andrew Beardsley MBE (born 18 January 1961[1]) is an English former footballer who played as a forward or midfielder between 1979 and 1999. In 1987, he set a record transfer fee in the English game and represented his country 59 times between 1986 and 1996, once as captain, taking part in two FIFA World Cups (1986 and 1990) and UEFA Euro 1988. At club level, he played for Newcastle United, Liverpool and Everton, having also had spells with Carlisle United, Manchester United, Vancouver Whitecaps, Bolton Wanderers, Manchester City, Fulham, Hartlepool United and the Melbourne Knights. He was briefly appointed as the caretaker manager of Newcastle United in 2010.",
            "query": "only player to play for manchester united manchester city liverpool and everton"
        },
        {
            "passage": "What's My Line? What's My Line? is a panel game show that originally ran in the United States on the CBS Television Network from 1950 to 1967, with several international versions and subsequent U.S. revivals. The game requires celebrity panelists to question a contestant in order to determine his or her occupation, i.e., \"line [of work],\" with panelists occasionally being called on to identify a celebrity \"mystery guest\" with specificity. It is the longest-running U.S. primetime network television game-show. Moderated by John Daly and with panelists Dorothy Kilgallen, Arlene Francis, and Bennett Cerf, What's My Line? won three Emmy Awards for \"Best Quiz or Audience Participation Show\" in 1952, 1953, and 1958 and the Golden Globe for Best TV Show in 1962.[1][2]",
            "query": "who was the original host of what's my line"
        }
    ],
    "dbpedia-entity": [
        {
            "passage": "Bourbonnais, Illinois Bourbonnais (pronounced /b\u028a\u0259rbo\u028a\u02c8ne\u026a/ or /b\u025cr\u02c8bo\u028an\u026as/) is a village in Kankakee County, Illinois, United States. The population was 15,256 at the 2000 census, but had grown to 18,631 in for the 2010 census. It is part of the Kankakee-Bourbonnais-Bradley Metropolitan Statistical Area and the Chicago\u2013Naperville\u2013Michigan City, IL-IN-WI Combined Statistical Area.",
            "query": "bourbonnais il"
        },
        {
            "passage": "History of Virginia Beach The history of Virginia Beach, Virginia, goes back to the Native Americans who lived in the area for thousands of years before the English colonists landed at Cape Henry in April 1607 and established their first permanent settlement at Jamestown a few weeks later.",
            "query": "city of virginia beach"
        },
        {
            "passage": "Austria Austria (/\u02c8\u0252\u02d0stri\u0259/; German: \u00d6sterreich [\u02c8\u00f8\u02d0st\u0250\u02cc\u0281a\u026a\u00e7]), officially the Republic of Austria (German: Republik \u00d6sterreich, About this sound listen ), is a federal republic and a landlocked country of over 8.5 million people  in Central Europe. It is bordered by the Czech Republic and Germany to the north, Hungary and Slovakia to the east, Slovenia and Italy to the south, and Switzerland and Liechtenstein to the west. The territory of Austria covers 83,879 square kilometres (32,386 sq mi).",
            "query": "Which countries have places with more than two caves?"
        }
    ]
}


def get_query_generation_prompt(dataset_name: str, passage: str, use_examples: bool) -> str:
    task = TASK_DICT[dataset_name]
    query_type = QUERY_TYPE_DICT[dataset_name]
    passage_type = PASSAGE_TYPE_DICT[dataset_name]
    if 'claim' in query_type:
        query_length = random.choice(CLAIM_QUERY_LENGTH_LIST)
    else:
        query_length = random.choice(QUERY_LENGTH_LIST)
    
    query_style = random.choice(QUERY_STYLE_LIST)
    
    if dataset_name in CLAIM_DATASET_LIST:
        mission = QG_MISSION_DICT[dataset_name].format(
        query_type=query_type, passage_type=passage_type,
        argu_type=random.choice(['supported', 'refuted'])
    )
    else:
        mission = QG_MISSION_DICT[dataset_name].format(
            query_type=query_type, passage_type=passage_type
        )

    # if dataset_name in EXAMPLE_DATASET_LIST:
    if use_examples is True and dataset_name in list(EXAMPLES_DICT.keys()):
        prompt_template = """\
Here is a retrieval task (Task) and a {passage_type} (Passage):

Task: {task}

Passage:
```
{passage}
```

Given the retrieval task and the {passage_type}, your mission is:
- {mission}

Note:
- The generated {query_type} should not contain the pronouns such as "this", "that", "it", "there", "here", etc.
- The generated {query_type} should be clear and {query_length}.
- The generated {query_type} should be {query_style} in terms of language style.

Here are a few examples for your reference:
1. Example 1:
Passage:
```
{example1_passage}
```
Generated query: {example1_query}

2. Example 2:
Passage:
```
{example2_passage}
```
Generated query: {example2_query}

3. Example 3:
Passage:
```
{example3_passage}
```
Generated query: {example3_query}

Your output should be a string of the generated {query_type}. Remember do not explain your output.

Your output:"""

        prompt = prompt_template.format(
            task=task,
            passage_type=passage_type,
            passage=passage,
            mission=mission,
            query_type=query_type,
            query_length=query_length,
            query_style=query_style,
            example1_passage=EXAMPLES_DICT[dataset_name][0]['passage'],
            example1_query=EXAMPLES_DICT[dataset_name][0]['query'],
            example2_passage=EXAMPLES_DICT[dataset_name][1]['passage'],
            example2_query=EXAMPLES_DICT[dataset_name][1]['query'],
            example3_passage=EXAMPLES_DICT[dataset_name][2]['passage'],
            example3_query=EXAMPLES_DICT[dataset_name][2]['query'],
        )
    else:
        prompt_template = """\
Here is a retrieval task (Task) and a {passage_type} (Passage):

Task: {task}

Passage:
```
{passage}
```

Given the retrieval task and the {passage_type}, your mission is:
- {mission}

Note:
- The generated {query_type} should not contain the pronouns such as "this", "that", "it", "there", "here", etc.
- The generated {query_type} should be clear and {query_length}.
- The generated {query_type} should be {query_style} in terms of language style.

Your output should be a string of the generated {query_type}. Remember do not explain your output.

Your output:"""

        prompt = prompt_template.format(
            task=task,
            passage_type=passage_type,
            passage=passage,
            mission=mission,
            query_type=query_type,
            query_length=query_length,
            query_style=query_style,
        )
    return prompt


ANSWER_TYPE_DICT = {
    'dbpedia-entity': 'entity description',
    'arguana': 'document',
    'climate-fever': 'answer',
    'cqadupstack': 'question description',
    'fever': 'answer',
    'fiqa': 'answer',
    'hotpotqa': 'answer',
    'msmarco': 'answer',
    'nfcorpus': 'answer',
    'nq': 'answer',
    'quora': 'question',
    'scidocs': 'paper abstract',
    'scifact': 'answer',
    'webis-touche2020': 'answer',
    'trec-covid': 'answer',
    'arxiv': 'answer',
    'news': 'answer',
    'finance': 'answer',
    'healthcare': 'answer',
    'law': 'answer'
}


def get_additional_info_generation_prompt(dataset_name: str, query: str) -> str:
    task = TASK_DICT[dataset_name]
    query_type = QUERY_TYPE_DICT[dataset_name]
    answer_type = ANSWER_TYPE_DICT[dataset_name]
    
    prompt_template = """\
Given a retrieval task and a query, your mission is to generate a brief {answer_type} for the query in the context of the retrieval task.
Please generate without any explanation.

Task: {task}

Query: {query}

Your output:"""

    prompt = prompt_template.format(
        task=task,
        query_type=query_type,
        query=query,
        answer_type=answer_type,
    )
    return prompt

def get_additional_info_generation_long_prompt(dataset_name: str, query: str) -> str:
    task = TASK_DICT[dataset_name]
    query_type = QUERY_TYPE_DICT[dataset_name]
    answer_type = ANSWER_TYPE_DICT[dataset_name]
    
    prompt_template = """\
Given a retrieval task and a query, your mission is to generate a {answer_type} about 100 words for the query in the context of the retrieval task.
Please generate without any explanation.

Task: {task}

Query: {query}

Your output:"""

    prompt = prompt_template.format(
        task=task,
        query_type=query_type,
        query=query,
        answer_type=answer_type,
    )
    return prompt

def get_additional_info_generation_long_air_prompt(dataset_name: str, query: str) -> str:
    task = TASK_DICT[dataset_name]
    query_type = QUERY_TYPE_DICT[dataset_name]
    answer_type = ANSWER_TYPE_DICT[dataset_name]
    
    prompt_template = """\
Given a retrieval task and a query, your mission is to generate a brief {answer_type} for the query in the context of the retrieval task.
Please generate without any explanation.

Task: {task}

Query: {query}

Your output:"""

    prompt = prompt_template.format(
        task=task,
        query_type=query_type,
        query=query,
        answer_type=answer_type,
    )
    return prompt


def get_additional_info_generation_train_prompt(dataset_name: str, query: str, reference: str) -> str:
    task = TASK_DICT[dataset_name]
    query_type = QUERY_TYPE_DICT[dataset_name]
    answer_type = ANSWER_TYPE_DICT[dataset_name]
    
    prompt_template = """\
Given a retrieval task, a query and a reference, your mission is to generate a brief {answer_type} for the query in the context of the retrieval task.
Please generate without any explanation.

Task: {task}

Query: {query}

Reference: {reference}

Your output:"""

    prompt = prompt_template.format(
        task=task,
        query_type=query_type,
        query=query,
        answer_type=answer_type,
        reference=reference
    )
    return prompt



QC_MISSION_DICT = {
    'dbpedia-entity': 'Judge whether the {passage_type} is relevant to the {query_type}.',
    'arguana': 'Judge whether the {passage_type} can refute the {query_type}.',
    'climate-fever': 'Judge whether the {passage_type} can support or refute the {query_type}.',
    'cqadupstack': 'Judge whether the {passage_type} is a duplicate to the given {query_type}.',
    'fever': 'Judge whether the {passage_type} can support or refute the {query_type}.',
    'fiqa': 'Judge whether the {passage_type} can answer the {query_type}.',
    'hotpotqa': 'Judge whether the {passage_type} can answer the {query_type}.',
    'msmarco': 'Judge whether the {passage_type} is relevant to the {query_type}.',
    'nfcorpus': 'Judge whether the {passage_type} can answer the {query_type}.',
    'nq': 'Judge whether the {passage_type} can answer the {query_type}.',
    'quora': 'Judge whether the {passage_type} is semantically equivalent to the given {query_type}.',
    'scidocs': 'Judge whether the {passage_type} is possibly cited by the paper with the given {query_type}.',
    'scifact': 'Judge whether the {passage_type} can support or refute the {query_type}.',
    'webis-touche2020': 'Judge whether the {passage_type} can answer the {query_type}.',
    'trec-covid': 'Judge whether the {passage_type} can answer the {query_type}.',
    'arxiv': 'Judge whether the {passage_type} can answer the {query_type}.',
    'news': 'Judge whether the {passage_type} can answer the {query_type}.',
    'finance': 'Judge whether the {passage_type} can answer the {query_type}.',
    'healthcare': 'Judge whether the {passage_type} can answer the {query_type}.',
    'law': 'Judge whether the {passage_type} can answer the {query_type}.',
}

QC_OPTION_DICT = {
    'dbpedia-entity': [
        'Yes, the {passage_type} is relevant to the {query_type}.',
        'No, the {passage_type} is not relevant to the {query_type}.',
    ],
    'arguana': [
        'Yes, the {passage_type} can refute the {query_type}.',
        'No, the {passage_type} cannot refute the {query_type}.',
    ],
    'climate-fever': [
        'Yes, the {passage_type} can support or refute the {query_type}.',
        'No, the {passage_type} can neither support nor refute the {query_type}.',
    ],
    'cqadupstack': [
        'Yes, the {passage_type} is a duplicate to the given {query_type}.',
        'No, the {passage_type} is not a duplicate to the given {query_type}.',
    ],
    'fever': [
        'Yes, the {passage_type} can support or refute the {query_type}.',
        'No, the {passage_type} can neither support nor refute the {query_type}.',
    ],
    'fiqa': [
        'Yes, the {passage_type} can answer the {query_type}.',
        'No, the {passage_type} cannot answer the {query_type}.',
    ],
    'hotpotqa': [
        'Yes, the {passage_type} can answer the {query_type}.',
        'No, the {passage_type} cannot answer the {query_type}.',
    ],
    'msmarco': [
        'Yes, the {passage_type} is relevant to the {query_type}.',
        'No, the {passage_type} is not relevant to the {query_type}.',
    ],
    'nfcorpus': [
        'Yes, the {passage_type} can answer the {query_type}.',
        'No, the {passage_type} cannot answer the {query_type}.',
    ],
    'nq': [
        'Yes, the {passage_type} can answer the {query_type}.',
        'No, the {passage_type} cannot answer the {query_type}.',
    ],
    'quora': [
        'Yes, the {passage_type} is semantically equivalent to the given {query_type}.',
        'No, the {passage_type} is not semantically equivalent to the given {query_type}.',
    ],
    'scidocs': [
        'Yes, the {passage_type} is possibly cited by the paper with the given {query_type}.',
        'No, the {passage_type} is not possibly cited by the paper with the given {query_type}.',
    ],
    'scifact': [
        'Yes, the {passage_type} can support or refute the {query_type}.',
        'No, the {passage_type} can neither support nor refute the {query_type}.',
    ],
    'webis-touche2020': [
        'Yes, the {passage_type} can answer the {query_type}.',
        'No, the {passage_type} cannot answer the {query_type}.',
    ],
    'trec-covid': [
        'Yes, the {passage_type} can answer the {query_type}.',
        'No, the {passage_type} cannot answer the {query_type}.',
    ],
    'arxiv': [
        'Yes, the {passage_type} can answer the {query_type}.',
        'No, the {passage_type} cannot answer the {query_type}.',
    ],
    'news': [
        'Yes, the {passage_type} can answer the {query_type}.',
        'No, the {passage_type} cannot answer the {query_type}.',
    ],
    'finance': [
        'Yes, the {passage_type} can answer the {query_type}.',
        'No, the {passage_type} cannot answer the {query_type}.',
    ],
    'healthcare': [
        'Yes, the {passage_type} can answer the {query_type}.',
        'No, the {passage_type} cannot answer the {query_type}.',
    ],
    'law': [
        'Yes, the {passage_type} can answer the {query_type}.',
        'No, the {passage_type} cannot answer the {query_type}.',
    ],
}


def get_quality_control_prompt(dataset_name: str, query: str, passage: str) -> str:
    """
    LLM 只输出 0 / 1，输出 0 表示 query 和 passage 在 task 下不相关，输出 1 表示在 task 下相关
    """
    task = TASK_DICT[dataset_name]
    query_type = QUERY_TYPE_DICT[dataset_name]
    passage_type = PASSAGE_TYPE_DICT[dataset_name]
    mission = QC_MISSION_DICT[dataset_name]
    pos_option = QC_OPTION_DICT[dataset_name][0].format(
        passage_type=passage_type, query_type=query_type
    )
    neg_option = QC_OPTION_DICT[dataset_name][1].format(
        passage_type=passage_type, query_type=query_type
    )
    
    prompt_template = """\
Given a retrieval task (Task), a {query_type} (Query), and a {passage_type} (Passage), your mission is {mission}.

Task: {task}

Query: {query}

Passage:
```
{passage}
```

Your output must be one of the following:
- 0: {neg_option}
- 1: {pos_option}

Do not explain your answer in the output. Your output must be a single number.

Your output:"""

    prompt = prompt_template.format(
        task=task,
        query_type=query_type,
        passage_type=passage_type,
        query=query,
        passage=passage,
        mission=mission,
        pos_option=pos_option,
        neg_option=neg_option,
    )
    return prompt


def get_reranker_prompt(dataset_name: str, query: str, passage: str) -> str:
    task = TASK_DICT[dataset_name]
    query_type = QUERY_TYPE_DICT[dataset_name]
    passage_type = PASSAGE_TYPE_DICT[dataset_name]
    
    prompt_template = """\
Given a retrieval task (Task), a {query_type} (Query), and a {passage_type} (Passage), your mission is to judge the relevance between the query and the passage in the context of the retrieval task from a scale of 0 to 4.

Task: {task}

Query: {query}

Passage:
```
{passage}
```

Your output must be a number between 0 and 4.

Your output:"""

    prompt = prompt_template.format(
        task=task,
        query_type=query_type,
        passage_type=passage_type,
        query=query,
        passage=passage,
    )
    return prompt


RERANKER_OPTION_SCORE_DICT = {
    "0": 10,
    "1": 20,
    "2": 30,
    "3": 40,
    "4": 50,
}