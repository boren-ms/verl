# Domain Keyword Reference

Baseline keyword lists for domain classification. Each domain needs 15-25 keywords.
A text is classified into a domain when **≥2 keywords** match (case-insensitive).

Extend or adapt these lists for your specific use case.

## Finance & Business

### Banking
`bank`, `banking`, `deposit`, `withdrawal`, `savings account`, `checking account`,
`loan`, `mortgage`, `credit`, `debit`, `interest rate`, `atm`, `wire transfer`,
`online banking`, `branch`, `fdic`, `treasury`, `central bank`, `federal reserve`,
`commercial bank`, `investment bank`, `fintech`, `cryptocurrency`

### CapitalMarket
`capital market`, `stock market`, `equity`, `bond`, `securities`, `trading`,
`investment`, `portfolio`, `hedge fund`, `mutual fund`, `ipo`, `dividend`,
`market cap`, `bull market`, `bear market`, `derivative`, `futures`, `options`,
`commodities`, `forex`, `financial analyst`, `wall street`, `nasdaq`, `dow jones`,
`s&p 500`, `asset management`, `venture capital`, `private equity`

### Insurance
`insurance`, `policyholder`, `premium`, `deductible`, `underwriting`, `claim`,
`coverage`, `actuary`, `actuarial`, `liability`, `indemnity`, `reinsurance`,
`insurer`, `insured`, `beneficiary`, `annuity`, `property insurance`, `casualty`,
`workers compensation`, `health plan`, `risk assessment`, `loss ratio`, `co-pay`,
`copay`, `coinsurance`

### Retail
`retail`, `shopping`, `store`, `purchase`, `consumer`, `product`, `e-commerce`,
`ecommerce`, `marketplace`, `inventory`, `merchandise`, `discount`, `coupon`,
`checkout`, `cart`, `order`, `shipping`, `customer service`, `return policy`,
`brand`, `wholesale`, `supply chain`, `point of sale`, `barcode`, `sku`

## Health & Medical

### LifeHealth
`health`, `wellness`, `nutrition`, `diet`, `exercise`, `fitness`, `mental health`,
`therapy`, `chronic disease`, `diabetes`, `cardiovascular`, `cancer`, `obesity`,
`prevention`, `public health`, `epidemiology`, `vaccine`, `immunization`,
`healthcare`, `life expectancy`, `mortality`, `morbidity`, `clinical trial`

### DoctorPatientConsultation
`doctor`, `physician`, `consultation`, `patient`, `diagnosis`, `treatment`,
`prognosis`, `referral`, `follow-up`, `appointment`, `medical advice`, `clinical`,
`outpatient`, `inpatient`, `hospital`, `emergency room`, `primary care`,
`specialist`, `surgeon`, `telemedicine`, `telehealth`, `medical examination`

### PatientHistoryDictation
`patient history`, `medical history`, `chief complaint`, `diagnosis`, `symptoms`,
`medication`, `prescription`, `allergy`, `vital signs`, `blood pressure`,
`heart rate`, `temperature`, `physical examination`, `medical record`,
`clinical notes`, `dictation`, `hpi`, `history of present illness`,
`past medical history`, `family history`, `surgical history`, `review of systems`,
`assessment and plan`

## Science & Technology

### ScienceTech
`science`, `technology`, `research`, `experiment`, `hypothesis`, `laboratory`,
`scientific`, `innovation`, `engineering`, `physics`, `chemistry`, `biology`,
`computer science`, `artificial intelligence`, `machine learning`, `algorithm`,
`software`, `hardware`, `robotics`, `quantum`, `nanotechnology`, `biotechnology`,
`genome`, `semiconductor`

### Energy
`energy`, `electricity`, `power plant`, `renewable`, `solar`, `wind energy`,
`hydroelectric`, `nuclear energy`, `fossil fuel`, `oil and gas`, `petroleum`,
`natural gas`, `coal`, `biomass`, `geothermal`, `grid`, `transmission`,
`kilowatt`, `megawatt`, `energy efficiency`, `carbon emission`, `utility`,
`turbine`, `photovoltaic`, `battery storage`

### Sustain (Sustainability)
`sustainability`, `sustainable`, `climate change`, `carbon footprint`,
`greenhouse gas`, `recycling`, `renewable energy`, `biodiversity`, `conservation`,
`ecosystem`, `environmental`, `green energy`, `circular economy`, `zero waste`,
`esg`, `carbon neutral`, `net zero`, `deforestation`, `pollution`, `clean energy`,
`sustainable development`, `paris agreement`

## Education & Media

### K12HigherEdu
`education`, `school`, `university`, `college`, `student`, `teacher`, `classroom`,
`curriculum`, `syllabus`, `lecture`, `professor`, `kindergarten`, `elementary`,
`middle school`, `high school`, `undergraduate`, `graduate`, `phd`, `dissertation`,
`thesis`, `tuition`, `scholarship`, `campus`, `semester`, `academic`, `enrollment`,
`gpa`, `homework`, `exam`, `degree`

### Media
`media`, `journalism`, `news`, `broadcast`, `television`, `radio`, `podcast`,
`streaming`, `social media`, `content creator`, `advertising`, `marketing`,
`public relations`, `entertainment`, `film`, `movie`, `documentary`, `newspaper`,
`magazine`, `digital media`, `influencer`, `viral`, `audience`, `ratings`

## Industry

### Manufactory (Manufacturing)
`manufacturing`, `factory`, `production`, `assembly`, `industrial`, `automation`,
`quality control`, `supply chain`, `lean manufacturing`, `six sigma`, `cnc`,
`machining`, `welding`, `fabrication`, `warehouse`, `logistics`, `iso 9001`,
`defect`, `tooling`, `injection molding`, `stamping`, `forging`, `casting`

### Gaming
`game`, `gaming`, `gamer`, `video game`, `esports`, `playstation`, `xbox`,
`nintendo`, `steam`, `twitch`, `fps`, `rpg`, `mmorpg`, `multiplayer`, `console`,
`gameplay`, `fortnite`, `minecraft`, `league of legends`, `overwatch`, `valorant`,
`call of duty`, `battlefield`, `apex legends`, `level up`, `quest`, `boss fight`,
`dungeon`, `loot`, `spawn`

## Adding New Domains

To add a domain:
1. Pick 15-25 keywords — mix of broad terms and domain-specific jargon
2. Include both full phrases and abbreviations
3. Test with a small sample to verify ≥2-hit threshold works
4. Add to the `DOMAIN_KEYWORDS` dict in `search_and_analyze.py`
