from clustering.docs_clustering import DocsClustering

model_name = input('Enter the model name that you want to perform clustering to it ? - antique or - wikipedia : ')
clustering = DocsClustering(model_name=model_name)
clustering.perform_clustering()
