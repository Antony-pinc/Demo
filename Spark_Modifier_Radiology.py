#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os

from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

from owlready2 import *
from owlready2.pymedtermino2 import *
from owlready2.pymedtermino2.umls import *

import pandas as pd
import spacy
from negspacy.negation import Negex
from negspacy.termsets import termset
from spacy.pipeline import merge_entities
import scispacy
import en_core_web_lg
import en_core_sci_lg
from spacy.language import Language
from spacy.pipeline import EntityRuler 
from spacy.language import Language
from spacy.matcher import Matcher
from spacy.tokens import Token, Doc, Span
from spacy.tokens import Doc
from spacy.tokens import Span
from spacy.util import filter_spans
from spacy import displacy
from scispacy.linking import EntityLinker

import srsly

import sagemaker
from sagemaker import get_execution_role
import sagemaker_pyspark


role = get_execution_role()

# Configure Spark to use the SageMaker Spark dependency jars
jars = sagemaker_pyspark.classpath_jars()

classpath = ":".join(sagemaker_pyspark.classpath_jars())

# See the SageMaker Spark Github repo under sagemaker-pyspark-sdk
# to learn how to connect to a remote EMR cluster running Spark from a Notebook Instance.
spark = (
    SparkSession.builder.config("spark.driver.extraClassPath", classpath)
    .master("local[*]")
    .getOrCreate()
)


# In[2]:


df_pyspark = spark.read.csv("s3a://pas-nlp-sagemaker/radiology_clob/Output_rad_hist_result/part-00013-result.csv",header=True,inferSchema=True)


# In[3]:


df_pyspark.show()


# In[4]:


get_ipython().run_cell_magic('time', '', 'df_pyspark = spark.read.csv("s3a://pas-nlp-sagemaker/radiology_clob/Output_rad_hist_result/part-00013-result.csv",header=True,inferSchema=True)')


# In[5]:


import boto3

s3 = boto3.resource('s3')
my_bucket = s3.Bucket('pas-nlp-sagemaker')
chunks=[]
for object_summary in my_bucket.objects.filter(Prefix="radiology_clob/Output_rad_hist_result/"):
    chunks.append(object_summary.key)
#chunks=chunks[847:]   
chunks


# In[6]:


df_pyspark2 = df_pyspark[["UMLS_Span","patient_key","Test_Name","Modifier_Result","Numeric_Result","Is_Negated","UMLS_CUI","SNOMED","ICD_Code"]]
df_pyspark2 = df_pyspark2.dropDuplicates(['Modifier_Result'])
merged_df = df_pyspark2.withColumn('Modifier_Result', regexp_replace('Modifier_Result', 'NULL', ''))
merged_df = merged_df.filter(merged_df.Modifier_Result != '') 
print((merged_df.count(), len(merged_df.columns)))


# In[14]:


from pyspark.sql.functions import col
result = pd.DataFrame()
for chunk in chunks:
    print(chunk)
    chunk_id="".join(re.findall(r"part-\d{5}", chunk))
    df_pyspark = spark.read.csv("s3a://pas-nlp-sagemaker/"+chunk,header=True,inferSchema=True)
    df_pyspark2 = df_pyspark[["UMLS_Span","patient_key","Test_Name","Modifier_Result","Numeric_Result","Is_Negated","UMLS_CUI","SNOMED","ICD_Code"]]
    #df_pyspark2= df_pyspark2.select([col(c).cast("string") for c in df_pyspark2.columns])
    ### Removing Duplicates and Null Values
    df_pyspark2 = df_pyspark2.dropDuplicates(['Modifier_Result'])
    df_new = df_pyspark2.withColumn('Modifier_Result', regexp_replace('Modifier_Result', 'NULL', ''))
    df_new = df_new.filter(df_new.Modifier_Result != '') 
    pandasDF = df_new.toPandas()
    print(pandasDF.shape)
    result = pd.concat([pandasDF, result])
    result = result.drop_duplicates(["Modifier_Result"])
    print("Result ", result.shape)
    
    #merged_df = merged_df.union(df_new)
    #print((merged_df.count(), len(merged_df.columns)))


# In[15]:


result.to_csv("Modifier_Radiology_Clob.csv")


# In[ ]:


result.head()


# In[ ]:


# df_pyspark.columns


# In[ ]:


get_ipython().run_cell_magic('time', '', 'df_pyspark2 = df_pyspark[["UMLS_Span","patient_key","Test_Name","Modifier_Result","Numeric_Result","Is_Negated","UMLS_CUI","SNOMED","ICD_Code"]]\n### Removing Duplicates and Null Values\ndf_pyspark2 = df_pyspark2.dropDuplicates([\'Modifier_Result\'])\ndf_new = df_pyspark2.withColumn(\'Modifier_Result\', regexp_replace(\'Modifier_Result\', \'NULL\', \'\'))\ndf_new = df_new.filter(df_new.Modifier_Result != \'\') \nprint((df_new.count(), len(df_new.columns)))\ndf_new[["Modifier_Result"]].show(10)')


# #### Load Ontology

# In[5]:


get_ipython().run_cell_magic('time', '', '#getting terminologies through pymedtermino\ndefault_world.set_backend(filename = "pym.sqlite3")\n#import_umls("/home/ec2-user/SageMaker/HeadCT/umls-2020AA-full.zip", terminologies = ["ICD10", "SNOMEDCT_US", "CUI"])\ndefault_world.save()\nPYM = get_ontology("http://PYM/").load()\nSNOMEDCT_US = PYM["SNOMEDCT_US"]\nICD10 = PYM["ICD10"]\nCUI = PYM["CUI"]\n#testing connection to ontology\nPYM.Concepts([ CUI["C0013604"]]) >> SNOMEDCT_US')


# In[3]:


df = spark.read.csv("Modifier_Radiology_Clob.csv",header=True,inferSchema=True)


# In[14]:


df_mod = pd.read_csv("Modifier_Radiology_Clob.csv")


# In[4]:


df.show()


# In[15]:


df_mod.head()


# In[6]:


#nlp = spacy.load("en_core_sci_lg")
nlp=spacy.load("/home/ec2-user/SageMaker/Spacyv3/ner-v3/output_model/model-best-ner-spacyv3")


# In[7]:


nlp.add_pipe("scispacy_linker", config={"linker_name": "umls", 'threshold' : 0.55}) 


# In[8]:


def cui_linker(text):
    dictn = {}
    lst_snomedid = []
    lst_snomeddesc = []
    docs = nlp(text)
    for len_doc in range(len(docs.ents)):
        entity = docs.ents[len_doc]
        linker = nlp.get_pipe("scispacy_linker")
        for umls_ent in entity._.kb_ents:
            try:
                cui_val = linker.kb.cui_to_entity[umls_ent[0]]
                text = str(list(CUI[cui_val[0]] >> SNOMEDCT_US)[0] if len(list(CUI[cui_val[0]] >> SNOMEDCT_US))>0 else '')

                snomed_id = re.search(r'\d+', str(text))
                snomed_id = str(text[snomed_id.span()[0]:snomed_id.span()[1]])
                lst_snomedid.append(snomed_id)
                snomed_txt = re.search(r'#',str(text))
                snomed_txt = text[snomed_txt.span()[1]:].lstrip().replace("\n","")
                lst_snomeddesc.append(snomed_txt)
            except:
                pass
    dictn = {"Modifier_Snomed_Id" : str(lst_snomedid), "Modifier_Snomed_Desc" : str(lst_snomeddesc)}
    return(dictn)


# In[9]:


df_cui = cui_linker("Severe benign Worsening")


# In[10]:


df_cui


# In[ ]:


get_ipython().run_cell_magic('time', '', 'from tqdm import tqdm\ndf_mod["Mod_Result"] = [cui_linker(i) for i in tqdm(df_mod["Modifier_Result"])]\ndf_mod = pd.concat([df_mod, df_mod["Mod_Result"].apply(pd.Series)], axis=1)')


# In[ ]:


df_mod.to_csv("Modifier_Snomed_Large3.csv")


# In[ ]:


df_mod.head()


# In[ ]:





# In[ ]:





# In[ ]:




