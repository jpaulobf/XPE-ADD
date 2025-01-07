from pyspark.sql import SparkSession
from pyspark.sql.functions import col, unix_timestamp, lit, lag, split, udf
from pyspark.sql.types import DoubleType
from pyspark.sql.window import Window
from math import radians, sin, cos, sqrt, atan2
import math

# Função para calcular a distância em km entre duas coordenadas (lat, lon) usando a fórmula de Haversine
def haversine(lat1, lon1, lat2, lon2):
    if None in [lat1, lon1, lat2, lon2]:  # Lida com valores nulos
        return None
    R = 6371  # Raio da Terra em km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))

# Registrar como UDF
haversine_udf = udf(haversine, DoubleType())

# Inicializar a sessão do Spark
spark = SparkSession.builder.appName("Fraud Detection").getOrCreate()

# Caminho do arquivo no bucket S3
input_path = "s3://your-bucket-name/compras_2k_fraudes.log"

# Carregar o arquivo de logs
df = spark.read.csv(input_path, header=True, inferSchema=True)

# Mostra o DF
df.show()

# Exibir o esquema dos dados
df.printSchema()

# Janela para verificar compras consecutivas por CPF
window_spec = Window.partitionBy("CPF").orderBy("Data_Hora")

# Converter a coluna de Data_Hora para o formato timestamp
df = df.withColumn("Data_Hora", unix_timestamp(col("Data_Hora"), "yyyy-MM-dd HH:mm:ss").cast("timestamp"))

# Calcular o tempo entre compras consecutivas
df = df.withColumn("Tempo_Entre_Compras", unix_timestamp(col("Data_Hora")) - lag(unix_timestamp(col("Data_Hora"))).over(window_spec))

# Adicionar flag para compras impossíveis com base na distância e tempo de compra
df = df.withColumn(
    "Risco_Fraude",
    (col("Compra_Virtual") == lit(False)) & (col("Tempo_Entre_Compras") < 3600)  # Compras físicas em cidades distantes no mesmo intervalo
)

# Mostra o DF
df.show()

# Calcular a distância entre as compras consecutivas (em km)
df = df.withColumn(
    "Distancia_Cidades",
    haversine_udf(
        col("latitude"),
        col("longitude"),
        lag(col("latitude")).over(window_spec),
        lag(col("longitude")).over(window_spec)
    )
)

# Condição de risco (compras físicas em cidades > 100 km no mesmo intervalo de tempo)
df = df.withColumn(
    "Suspeita_Fraude",
    (col("Distancia_Cidades") > 100) & (col("Tempo_Entre_Compras") < 3600)  # Distância maior que 100km e intervalo menor que 60min
)

# Filtrar compras suspeitas
fraudes = df.filter(col("Suspeita_Fraude") == lit(True))

# Salvar resultado em um bucket S3
output_path = "s3://your-bucket-name/suspect_purchases/"
fraudes.write.csv(output_path, header=True, mode="overwrite")

# Exibir as fraudes encontradas
fraudes.show(truncate=False)

# Encerrar sessão do Spark
spark.stop()
