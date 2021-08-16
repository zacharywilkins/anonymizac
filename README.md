# anonymizac
This repo contains an initial Conversational AI data anonymization implementation by Zachary Wilkins. It could in theory be used as a starting point for removing PII from customers' data to ensure data security compliance, without compromising the utility of that data. 

## Why anonymize data?
Anonymization of customer data is a legal necessity for virtually every Conversational AI company. Conversational AI companies are often eager to train models on their customers' data. However, before that data is received, Conversational AI companies usually must comply with GDPR, SOC 2, and other security regulations. Conversational AI companies that do not comply may find that customers will refuse to work with the Conversational AI company out of fear of a user data privacy lawsuit. Most of these agreements require that data is anonymized before it is transferred from the customer to the Conversational AI service provider.

Anonymization techniques can ensure that Personally Identifiable Information ("PII") is removed from language data. PII is often defined as data that is specific enough to be tied to a specific individual. For example, say a dataset provided by a third party includes customer service requests, such as "Hello, I am having an issue with my laptop overheating. My laptop is Model 0539-40586-A and is a 2016 Macbook Air. Feel free to call me at (555) 943-4950 or email me at john@gmail.com". Here, the PII are the user's phone number and email, which a nefarious actor could use to figure out who the person is for their own personal gain. Examples such as the one I just gave would need to have PII such as phone numbers and emails removed before the data is sent to the AI service provider to funnel into their ETL for model training, testing, and deployment.

Effective anonymization should not only remove PII, but it should also ensure that non-PII data that can be useful for model training is not removed. For example, a poorly designed anonymization algorithm might remove all integers in an attempt to scrub phone numbers, social security numbers, etc. Unfortunately, the data would become far less useful if anonymized in this way. Numbers such as error codes, laptop model numbers, or the year the laptop is from could all be useful information to say, for example, train a model that identifies a user's specific IT issue. That data would have been erroneously removed as part of unsophisticated anonymization approaches, rendering the data devoid of certain high-utility pieces of information. 

## The data
As I believe should always be the case with Conversational AI, you need to start by first finding the closest relevant data possible for the problem you are trying to solve, ideally in the domain of your product. In this case, I focused on the domain of user-generated customer service requests. I found unanonymized data from an open-source dataset released by Amazon (http://jmcauley.ucsd.edu/data/amazon/qa/). See below a sample of that data that I compiled which contains PII (to be clear, this is readily available on the internet, so I am not including personal data that one cannot find through a simple series of Google searches).

```
# PII Examples from Amazon's `qa_Tools_and_Home_Improvement` dataset: 
pii_data = {
  "examples": [
    "Hi, I always say, if you do not know, ask. Yes, it is a plug in unit. It has th ree prongs. If you do not have three prong outlets you can buy adapters at the hardware st ore, they are very inexpensive. I bought 4 of the Lifesmart heaters last winter. I have a 2,700 sq ft house. In winter, I use 2 of them most the time and occasionally 3 and my hou se stays toasty warm! I live in Seattle. I am sure if you lived in a colder climate, say t he mid West a 2700 sq ft home would require 4 units. I was at the local hardware store tod ay. They have comparable units on sale for $249.00 each. Lifesmart units, look great and d o a great job! I bought 4 of them for $400.00. I see the price went up about $10.00 since I bought mine. Still a great price and the best price I have seen. Also, Lifesmart Deluxe Units come with a 5 year warranty and other units only come with a 1 year warranty. Of cou rse if you want to buy the EdenPure brand, the top of the line unit comes with a 5 year wa rranty, however, you will have to pay $399.00 per unit to get a 5 year warranty with EdenP ure . I bought 4 Lifesmart units for the price of 1 EdenPure top of the line unit and have a 5 year warranty on all my units. I love my Lifesmart Delux Stealth Series Infrared Porta ble Heaters They Are Awesome!!... Kate",
    "Can the cans be safely stored in my car? I live in Scottsdale Arizona and car t emps hit 150ish in the summer.",
    "Yes...get it .Well worth the money.I live in Canton,Ohio and has worked well al l winter ,and has a remote to control the brightness .We keep it low at night ,then when w e're outside with friends we turn it up.Great buy.",
    "Hello, do you have any of the other senco models on hand (16g) i live in montre al canada and other sellers do not seem to want to ship to us canucks",
    "I live in Gainesville Florida, is a thermostat recommended??",
    ...
  ]
}
```

As you can see, users have included a great deal of personal information that could make it possible to locate them in the world. Take the first user input as an example. The user includes the size of her home `I have a 2,700 sq ft house`, the city where she lives `I live in Seattle`, and her name `Kate`. To comply with security measures such as GDPR and SOC 2, and most B2B service agreements in the Conversational AI industry, it is sufficient to anonymize data to the point that a single individual cannot be identified from their data. 


## Scope
To the end of a simple initial implementation, I decided to isolate *location* as the NER (Named Entity Recognition) type that I would "scrub" or remove from the data in order to anonymize it. I will do this using a combination of linguistics heuristics and spaCy's en_core_web_md CNN (Convolutional Neural Network) model, which was trained with GloVe vectors on Common Crawl data. This is a multi-task model that provides useful NLP data such as part of speech tags, syntactic dependency parse data, lemmatized forms of the words, and so on. The goal would be, as in the example above, to take a sentence such as I live in Seattle and replace it with I live in [LOCATION] . That way, the data is anonymized and compliant from a data privacy perspective. Later on, if we find the [LOCATION] string to be detrimental to model training, we can replace it with some generic name, e.g. `Citysville`, or randomly replace [LOCATION] with some other city, e.g. `New York` or `London`. 

## Implementation overview
First, I decided to create a base `Anonymizer` class (`anonymizers/base_anonymizer.py`) that can be used as a starting point for anonymizing the data. This base class allows us to initialize the spaCy model, normalize the user inputs, and establishes an anonymization pipeline that we can use to remove PII from the data. Next, I built a child class I named `LocationAnonymizer`, which inherits all of the useful methods I defined in the base `Anonymizer` class. This class contains the location-specific scrubbing logic that we can use to scrub the data from Amazon that contains PII.

Let's test the implementation out on an example user input:

```
print(LocationAnonymizer().scrub("I currently live in an apartment in San Francisco with my cat Kieran")) 
I currently live in an apartment in [LOCATION] [LOCATION] with my cat Kieran
```

It works! Now, let's try out out our implementation on the full set of Amazon data included above. 

## Results
Let's compare before and after anonymization, starting with some shorter examples: 

Before:
```
#9 I can not answer the question as I live in Austin, TX. My unit is installe d inside the home that has an annual temperature variation from 68 degrees in the winter to 80 degrees in the summer. The unit has functioned flawlessly si nce I installed it. I am very happy with the unit. My original unit lasted 12 years plus. 
#17 Can the cans be safely stored in my car? I live in Scottsdale Arizona and car temps hit 150ish in the summer. 
#19 Hello, do you have any of the other senco models on hand (16g) i live in montreal canada and other sellers do not seem to want to ship to us canucks 
#20 I live in Gainesville Florida, is a thermostat recommended?? 
```

After:
```
#9 I can not answer the question as I live in [LOCATION], TX. My unit is inst alled inside the home that has an annual temperature variation from 68 degree s in the winter to 80 degrees in the summer. The unit has functioned flawless ly since I installed it. I am very happy with the unit. My original unit last ed 12 years plus. 
#17 Can the cans be safely stored in my car? I live in [LOCATION] [LOCATION] and car temps hit 150ish in the summer. 
#19 Hello, do you have any of the other senco models on hand (16 g) i live in [LOCATION] [LOCATION] and other sellers do not seem to want to ship to us canucks 
#20 I live in [LOCATION] [LOCATION], is a thermostat recommended??
```

Similarly, with our lengthy "Kate from Seattle" example, `I live in Seattle` was anonymized to `I live in [LOCATION]`, and nothing else was changed about the user input -- maintaining integrity of the quality of the training data.

