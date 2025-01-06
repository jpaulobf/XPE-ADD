from pyspark.sql import SparkSession

# Criação da sessão Spark
spark = SparkSession.builder.appName("EMR_LogAnalysis").getOrCreate()

# Lê o arquivo de log do S3
log_file = "s3://<SEU_BUCKET>/logs/access.log"  # Substitua pelo caminho real
logs_rdd = spark.sparkContext.textFile(log_file)

# Processa cada linha do log
ip_counts = (
    logs_rdd.map(lambda line: line.split(" ")[0])  # Extrai o IP (primeiro campo)
    .map(lambda ip: (ip, 1))  # Cria pares (IP, 1)
    .reduceByKey(lambda a, b: a + b)  # Soma os acessos por IP
)

# Converte para DataFrame para visualização
ip_counts_df = ip_counts.toDF(["IP", "Count"]).orderBy("Count", ascending=False)

# Salva os resultados no S3
output_path = "s3://<SEU_BUCKET>/results/ip_counts"
ip_counts_df.write.csv(output_path, header=True)

# Mostra os resultados (limitado para evitar grandes volumes)
ip_counts_df.show(10)

# Finaliza a sessão Spark
spark.stop()
