from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
import tiktoken
from ner_models import flair_model

def transcript_split(transcript):
    #GPT has a token limit of 4097 per message
    #This function splits the transcript into chunks ensuring that this limit is not exceeded.
    #The +250 condition states that, if the current number of tokens per chunk plus 250 would exceed 4097, then we have n+1 chunks.
    #This is just to ensure that we can fit in instructions and full sentences.
    #Sentences are not cut off in the middle, but included in full in each chunk.
    tokens = []
    for sentence_dict in transcript:
        tokens.append(len(tiktoken.get_encoding("cl100k_base").encode(sentence_dict["sentence"])))
    n = sum(tokens) // 4097 + 1
    if sum(tokens) / n + 250 > 4097:
        n+=1
    threshold = sum(tokens) / n
    chunks = []
    t = ''
    tokens = []
    for sentence_dict in transcript:
        tokens.append(len(tiktoken.get_encoding("cl100k_base").encode(sentence_dict["sentence"])))
        t += sentence_dict["sentence"]
        if sum(tokens) > threshold:
            chunks.append(t)
            t = ''
            tokens = []
    chunks.append(t)
    return chunks


def call_chat(response_schemas, prompt_template, args):
    #the langchain package was very useful for improving the consistency of GPT's responses.
    #see the rationale of what I have tried to do here:
    #https://medium.com/m/callback/google#state=google-%7Chttps://towardsdatascience.com/use-langchains-output-parser-with-chatgpt-for-structured-outputs-cf536f692685?source%253Dlogin--------------------------post_regwall-----------%2526skipOnboarding%253D1%7Clogin&access_token=ya29.a0AbVbY6N6QUHvR8gkzhJID6zh7VafFCpAaQFMBwZveKBIZGUCGjXEDoIRzODRFL1QUu7XZuf58G9JkCXz_djSvIq3znTmbuEPzk1XraxniQKWFmJeNJ5U7n6Azzxq9ZMTAQizmYuOBnZDqgfhTNGIZ_OEVRmu4YoaCgYKAXISARISFQFWKvPlPTte9B_AI8sGZcVNvmpTOA0166&token_type=Bearer&expires_in=3599&scope=email%20profile%20https://www.googleapis.com/auth/userinfo.profile%20https://www.googleapis.com/auth/userinfo.email%20openid&id_token=eyJhbGciOiJSUzI1NiIsImtpZCI6ImEzYmRiZmRlZGUzYmFiYjI2NTFhZmNhMjY3OGRkZThjMGIzNWRmNzYiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJhY2NvdW50cy5nb29nbGUuY29tIiwiYXpwIjoiMjE2Mjk2MDM1ODM0LWsxazZxZTA2MHMydHAyYTJqYW00bGpkY21zMDBzdHRnLmFwcHMuZ29vZ2xldXNlcmNvbnRlbnQuY29tIiwiYXVkIjoiMjE2Mjk2MDM1ODM0LWsxazZxZTA2MHMydHAyYTJqYW00bGpkY21zMDBzdHRnLmFwcHMuZ29vZ2xldXNlcmNvbnRlbnQuY29tIiwic3ViIjoiMTA0NzY5MzYyNjc5MzI3MDg1MTc5IiwiZW1haWwiOiJlbGxpb3RjaHJpc3RvcGhlcnNAZ21haWwuY29tIiwiZW1haWxfdmVyaWZpZWQiOnRydWUsImF0X2hhc2giOiJDSm81ZmNlMzE3XzdEcVg1a0YtTGRRIiwibm9uY2UiOiJmNjc1MzE2NjhhZmQwZTdkYTdkMDE0MWFmZTRhOTZlMDNjNzI0NTIyNDcyNTMzNDlhNmI1YjE2MjQ3MWM4ODcxIiwibmJmIjoxNjg5NDk5ODM5LCJpYXQiOjE2ODk1MDAxMzksImV4cCI6MTY4OTUwMzczOSwianRpIjoiOTNkNjEwZmI4MGU4YzdmODcxMDgwYzM4YjFlMTRiZTMxODBhMWUwZiJ9.DEAjdjBE_9r5fc1pL3p3mgtuKCQlvNhTl2wSJMrzRFaR8t-D6UCZbvQhybXoIWQzdXmrU2fp_ZnkkVjS0-PqH7wbA2pMyNfRf5Y_MaOqjIQaWQcXN5jDIMtGxyqHTMqkEz2BEw4NuF0Y7e-zdCRsr9DKEFVOUkKyJwsUjHuPlA60_1gb0atX3R34ptCBJjoV1ry_lfYhxPWxTIvzCIJar7mMkvBHCI4UtUxjEutDyD7k0XrdPcFnCJlpRz1AJsrVKly1CIAG7N8Vkx4SAlp4_4g5-DkxTy_142B9IwhFAe4SfRLtksdVWNwDqoukFY-BEo9UyhZXIHu7I3gLUscGsg&authuser=0&prompt=none
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    prompt_template = ChatPromptTemplate.from_template(template=prompt_template)
    #This is just from a free trial. 
    #There are restrictions on how often one can prompt GPT, which increases the run time as it will continue trying until allowed.
    my_key = "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"
    chat = ChatOpenAI(openai_api_key=my_key, temperature=0.0)
    messages = prompt_template.format_messages(
    arg0=args[0],
    arg1=args[1],
    format_instructions=format_instructions)
    response = chat(messages)
    return output_parser.parse(response.content)


def get_products(transcript):
    products_schema = ResponseSchema(
    name="products",
    description="Extract any products that are mentioned. A product is 'a clearly-defined, company-specific article, substance or system that is manufactured, compiled or refined specifically for sale'. We are only interested in products provided by the main company in the text. A good question to ask is 'Does the firm sell this product?' Please be as accurate as possible and really emphasise avoiding false positives: labelling something as a product when it is not. Respect the definition. \
    Return a list with all identified products."
)

    prompt_template = """From the given text, extract the following information:

products: Extract any products that are mentioned. A product is 'a clearly-defined, company-specific article, substance or system that is manufactured, compiled or refined specifically for sale'. We are only interested in products provided by the main company in the text. Please be as accurate as possible and really emphasise avoiding false positives: labelling something as a product when it is not. Respect the definition. \
Return a list with all identified products.

Format the output as JSON with the following key:
products

This JSON format is important.

Use the following, identified as products by a NER model, as a guide:
NER products: {arg0}

text: {arg1}
"""

    chunks = transcript_split(transcript)
    products = []
    flair_products = flair_model(transcript)
    i = 0
    while i < len(chunks):
        #the try-except block is important. even with the langchain format, GPT sometimes gives randomly-structured responses nonetheless
        #with try except, we simply loop until it does give the correct response structure. we print the error so that we can see if there is an actual issue or not
        try:
            output_dict = call_chat([products_schema], prompt_template, [flair_products, chunks[i]])
            p = output_dict.get('products')
            for q in p:
                products.append(q)
            i += 1
        except Exception as e:
            print(e)
    return list(set(products))


#we use the products identified to then find new products and name variations
def np_nv(transcript, products):
    new_products_schema = ResponseSchema(
    name="new_products",
    description="For the products, identify whether they are new releases or not, whether they are new products or not. \
Return a list with all identified new products."
)

    name_variations_schema = ResponseSchema(
    name="name_variations",
    description="Identify all name variations. Name variations are different ways in which the same product is referred to. For instance, if Product A is referred to as both X and Y, then X and Y are name variations. I want you to find all name variations for each product. It is possible that two products in the product list are in fact name variaitons for one another; in this case, include them only once. \
Return a list of lists with all identified name variations"
)


    prompt_template = """I will give you a list of products and a text. The list of products consists of entities that have been identified as product type by a named entity recognition process. I want you to consider these products when reading the text and find extra information about them. Please be as accurate as possible and really emphasise avoiding false positives if you are uncertain about including something, err on the side of not including it.
Due to constraints on how many tokens I can send in a message, I will split the text into multiple chunks. This means that it is highly likely that some of the products in the list will not be present in the text seen in this message. This is fine and expected, simply consider the products present in the text, and from the text extract the following information about those products:

new_products: For the products, identify whether they are new releases or not, whether they are new products or not.  \
Return a list with all identified new products.

name_variations: Identify all name variations. Name variations are different ways in which the same product is referred to. For instance, if Product A is referred to as both X and Y, then X and Y are name variations. I want you to find all name variations for each product. It is possible that two products in the product list are in fact name variaitons for one another; in this case, include them only once. \
Return a list of lists with all identified name variations.

Format the output as JSON with the following keys:
new_products
name_variations

This JSON format is important

The following is the list of products in the text.
products: {arg0}

text: {arg1}
"""

    chunks = transcript_split(transcript)
    new_products = []
    name_variations = []

    i = 0
    while i < len(chunks):
        try:
            output_dict = call_chat([new_products_schema, name_variations_schema], prompt_template, [products, chunks[i]])
            np = output_dict.get('new_products')
            for pn in np:
                new_products.append(pn)
            nv = output_dict.get('name_variations')            
            for vn in nv:
                name_variations.append(vn)
            i += 1
        except Exception as e:
            print(e)
    name_variations = [x for x in name_variations if len(x) > 1]
    return list(set(new_products)), name_variations


#the name_variations are highly unstructured, as a result of having to ask GPT for name variations from several chunks of the same text
#therefore, the same name variations have often been included multiple times
#the following simply prompts GPT outright about the list of name variations, asking it to clean the data and, if very obvious errors have occured, to correct those
def nv_monitoring(products, name_variations):
    name_variations_schema = ResponseSchema(
    name="name_variations",
    description="""Below I have given you a list of lists of name variations. Name variations are different ways in which the same product is referred to. For instance, if Product A is referred to as both X and Y, then X and Y are name variations. The name variations below have been generated by a named entity recognition process. I want you to analyse the name variations and, if it is necessary, change the list.
For instance, the list has been produced by iterating over different sections of a piece of text and then appending the name variations found in each section to a common list. As such, it is quite likely that the same name variations are to be found in multiple places in the list. These may not necessarily be ordered in the same way or have the identical variations, but they are name variations for the same product. Where this is obviuos, change the list so that there is only one entry for this product.
It is also possible that the NER model has produced a set of name variations which are, in fact, not name variations of one another. This will obviously be difficult for you to identify without the context of the text from which they were generated, but if it is an obvious error, perhaps given your knowledge of the world up until your knowledge cutoff, then do remove this set of name variations as well.
Note that, in most cases, there will be name variations. If you are inclined to return an empty list, you should definietely reconsider. If I give you a list with several potential name variations, you HAVE TO return some name variations. \

Here are a couple of examples (nv -> name variation, p -> product):
input: [['nv1_p1', 'nv2_p1'], ['nv2_p1', 'nv3_p1']]
output: [['nv1_p1', 'nv2_p1', 'nv3_p1']]
There is only one product (p1), but it has name variations in two different lists. Thus, we only return one list for this product. Also note that p1 has three name variations, taken from the two different lists.

input: [['nv1_p1', 'nv1_p2'], ['nv1_p3', 'nv2_p3']]
output: [['nv1_p3', 'nv2_p3']]
Here the first list has name variations but for two different products, meaning we remove it. The second list has name variations for the same product, so we keep it.

Combining these two examples:
input: [['nv1_p1', 'nv2_p1'], ['nv2_p1', 'nv3_p1'], ['nv1_p1', 'nv1_p2'], ['nv1_p3', 'nv2_p3']]
output: [['nv1_p1', 'nv2_p1', 'nv3_p1'], ['nv1_p3', 'nv2_p3']]
Note that, although we see p2 in the input, we do not see it in the output, since its inclusion as a nv was invalid (different products) and it has no real name variations elsewhere.

Return a list of lists with all correct name variations.

Do NOT return an empty list!"""
)
    prompt_template = """Below I have given you a list of lists of name variations. Name variations are different ways in which the same product is referred to. For instance, if Product A is referred to as both X and Y, then X and Y are name variations. The name variations below have been generated by a named entity recognition process. I want you to analyse the name variations and, if it is necessary, change the list.
For instance, the list has been produced by iterating over different sections of a piece of text and then appending the name variations found in each section to a common list. As such, it is quite likely that the same name variations are to be found in multiple places in the list. These may not necessarily be ordered in the same way or have the identical variations, but they are name variations for the same product. Where this is obviuos, change the list so that there is only one entry for this product.
It is also possible that the NER model has produced a set of name variations which are, in fact, not name variations of one another. This will obviously be difficult for you to identify without the context of the text from which they were generated, but if it is an obvious error, perhaps given your knowledge of the world up until your knowledge cutoff, then do remove this set of name variations as well.
Note that, in most cases, there will be name variations. If you are inclined to return an empty list, you should definietely reconsider. If I give you a list with several potential name variations, you HAVE TO return some name variations. \

Here are a couple of examples (nv -> name variation, p -> product):
input: [['nv1_p1', 'nv2_p1'], ['nv2_p1', 'nv3_p1']]
output: [['nv1_p1', 'nv2_p1', 'nv3_p1']]
There is only one product (p1), but it has name variations in two different lists. Thus, we only return one list for this product. Also note that p1 has three name variations, taken from the two different lists.

input: [['nv1_p1', 'nv1_p2'], ['nv1_p3', 'nv2_p3']]
output: [['nv1_p3', 'nv2_p3']]
Here the first list has name variations but for two different products, meaning we remove it. The second list has name variations for the same product, so we keep it.

Combining these two examples:
input: [['nv1_p1', 'nv2_p1'], ['nv2_p1', 'nv3_p1'], ['nv1_p1', 'nv1_p2'], ['nv1_p3', 'nv2_p3']]
output: [['nv1_p1', 'nv2_p1', 'nv3_p1'], ['nv1_p3', 'nv2_p3']]
Note that, although we see p2 in the input, we do not see it in the output, since its inclusion as a nv was invalid (different products) and it has no real name variations elsewhere.

Return a list of lists with all correct name variations.

Do NOT return an empty list!

Format the output as JSON with the following key:
name_variations

This JSON format is important

The following is the list of lists of name variations that I want you to adjust.
name_variations: {arg0}

Here is a list of the products for which there exist name variations in the list above. It is possible that products in this list have been identified as name variations for one another in the list above.
products: {arg1}
"""

    while True:
        try:
            output_dict = call_chat([name_variations_schema], prompt_template, [name_variations, products])
            name_variations = output_dict.get('name_variations')
            name_variations = [x for x in name_variations if len(x) > 1]
            return name_variations
        except Exception as e:
            print(e)


#This is where we run the code
import json
files = ['aapl_q1_2023', 'abb_q3_2022', 'abbv_q3_2022', 'abt_q4_2022']
file = 0
#I had my files in a texelio subfolder. Obviously just replace the path.
with open(fr'{files[file]}.json') as f:
    data = json.load(f)
transcript = data["transcript"]

#Here we iterate over the get_products function n times and then combine the result, hoping to improve the recall
products_n_times = 3
products = []
for i in range(products_n_times):
    print(i+1)
    p = get_products(transcript)
    for q in p:
        products.append(q)
    print(products)
products = list(set(products))
print(products)

new_products, name_variations = np_nv(transcript, products)

print(products)
print(new_products)
print(name_variations)


if len(name_variations) > 0:
    name_variations = nv_monitoring(products, name_variations)

print(f'Products: {products}\nnNew Products: {new_products}\nName Variations: {name_variations}')