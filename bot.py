import nltk
from nltk.chat.util import Chat, reflections
import spacy
from difflib import get_close_matches


# Intentar cargar el modelo de spaCy para español
try:
    nlp = spacy.load("es_core_news_sm")
except:
    print("El modelo de spaCy para español no está instalado. Ejecuta 'python -m spacy download es_core_news_sm' para instalarlo.")
    exit()

# Datos de la empresa y sus terminales
datos_empresa = {
    "Internacional de Contenedores Asociados de Veracruz": {
        "telefono": "+52 (229) 989.5400",
        "correo": "billing.icave@hutchisonports.com.mx"
    },
    "Lázaro Cárdenas Terminal Portuaria de Contenedores": {
        "telefono": "+52 (75) 3533.0500",
        "correo": "billing.lctpc@hutchisonports.com.mx"
    },
    "Terminal Internacional de Manzanillo": {
        "telefono": "+52 (314) 331.2701/02",
        "correo": "billing.timsa@hutchisonports.com.mx"
    },
    "Ensenada International Terminal": {
        "telefono": "+52 (646) 178-8801 al 03",
        "correo": "billing.eit@hutchisonports.com.mx"
    }
}

# Preguntas frecuentes
faqs = [
    {
        "pregunta": "¿Cuál es el modo de transporte más apropiado para exportar productos perecederos?",
        "respuesta": "Para productos perecederos, el transporte aéreo suele ser el más recomendado debido a su rapidez y la capacidad de mantener la cadena de frío durante el tránsito. Sin embargo, también se puede considerar el transporte marítimo refrigerado para ciertos productos y destinos, evaluando factores como el tiempo de tránsito y los costos."
    },
    {
        "pregunta": "¿Qué modalidad de transporte puedo utilizar para enviar mercancía a Europa?",
        "respuesta": "Para envíos a Europa, las opciones más comunes son el transporte marítimo y el transporte aéreo. El transporte marítimo es más económico pero toma más tiempo, mientras que el transporte aéreo es más rápido pero más costoso. La elección dependerá del tipo de mercancía, los plazos de entrega y el presupuesto disponible."
    },
    {
        "pregunta": "¿Cómo decido el mejor modo de transporte para mi producto?",
        "respuesta": "Para decidir el mejor modo de transporte para tu producto, debes considerar factores como el tipo de producto (perecedero, frágil, peligroso, etc.), el peso y volumen, el valor de la mercancía, los plazos de entrega requeridos, el destino y la distancia, así como tu presupuesto. Realizar un análisis cuidadoso de estos factores te permitirá elegir la opción más adecuada, ya sea transporte marítimo, aéreo, terrestre o una combinación de ellos."
    },
    {
        "pregunta": "¿Cuál es el valor del flete para el transporte internacional (marítimo)?",
        "respuesta": "El valor del flete marítimo depende de varios factores, como el peso, el volumen, el tipo de carga, el puerto de origen y destino, entre otros. Para obtener una cotización precisa, te recomiendo comunicarte con una empresa de transporte marítimo o un agente de carga."
    },
    {
        "pregunta": "¿Qué documentación se requiere para exportar a Estados Unidos?",
        "respuesta": "Los documentos comúnmente requeridos para exportar a Estados Unidos son: factura comercial, lista de empaque, conocimiento de embarque, certificado de origen, y cualquier documento adicional específico para el producto. Es recomendable verificar los requisitos actualizados con las autoridades correspondientes."
    },
    {
        "pregunta": "¿Cuáles son los requisitos para el embalaje de productos frágiles?",
        "respuesta": "Para el embalaje de productos frágiles, se recomienda utilizar materiales de amortiguación como espumas, burbujas de plástico o papel de relleno. También es importante etiquetar correctamente los paquetes como 'Frágil' y manejarlos con cuidado durante el transporte y la carga/descarga."
    },
    {
        "pregunta": "¿Cuál es el modo de transporte más apropiado para mi producto?",
        "respuesta": ("La definición del modo de transporte para exportar un producto está ligada a diferentes factores entre ellos: "
                      "clase, valor, peso y volumen del producto; tipo de carga; manipuleos; clase y costo de embalaje; seguridad; "
                      "accesibilidad, disponibilidad, frecuencia, rapidez y costo del modo de transporte, adicionalmente el costo de aduana "
                      "y la documentación exigida podrán incidir en la decisión de utilizar los modos marítimos, aéreos o terrestres. En Perú "
                      "existe una amplia gama de servicios marítimos y aéreos principalmente y terrestres al resto de los países vecinos. El transporte "
                      "para algunos países se ve limitado por las frecuencias y los tiempos de tránsito que se toman para llegar al destino final.")
    }
]

# Función para encontrar la respuesta a partir de entidades nombradas
def get_info_based_on_entity(text):
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "ORG":  # Organización puede ser usado para nombres de terminales
            terminal_encontrada = ent.text
            terminales = list(datos_empresa.keys())
            coincidencias = get_close_matches(terminal_encontrada, terminales, n=1, cutoff=0.8)
            if coincidencias:
                terminal = coincidencias[0]
                return lambda: "El número de teléfono de {0} es {1} y el correo de facturación es {2}".format(terminal, datos_empresa[terminal]["telefono"], datos_empresa[terminal]["correo"])
    
    # Buscar coincidencias aproximadas en las preguntas frecuentes
    preguntas = [faq["pregunta"].lower() for faq in faqs]
    coincidencias = get_close_matches(text.lower(), preguntas, n=1, cutoff=0.8)
    if coincidencias:
        for faq in faqs:
            if faq["pregunta"].lower() == coincidencias[0]:
                return lambda: faq["respuesta"]
    
    # Buscar patrones de preguntas sobre modos de transporte
    transporte_patterns = [
        "modo de transporte",
        "transporte apropiado",
        "modalidad de transporte",
        "enviar mercancía"
    ]
    for pattern in transporte_patterns:
        if pattern in text.lower():
            return lambda: faqs[0]["respuesta"]  # Devolver la respuesta genérica sobre modos de transporte
    
    # Buscar patrones de preguntas sobre servicios adicionales
    servicio_patterns = [
        "seguro de transporte",
        "envío de mercancía"
    ]
    for pattern in servicio_patterns:
        if pattern in text.lower():
            return lambda: "Lo siento, no ofrecemos servicios adicionales como seguros de transporte o envíos directos de mercancía. Puedo proporcionarte información sobre nuestras terminales y modos de transporte."
    
    return lambda: "Lo siento, no tengo información sobre eso. Puedes preguntar sobre alguna de las terminales, modos de transporte o hacerme preguntas frecuentes."

# Clase personalizada que hereda de Chat
class MiChat(Chat):
    def respond(self, text):
        get_info_func = get_info_based_on_entity(text)
        response = get_info_func()
        return response

# Función para manejar la conversación
def chatbot():
    print("¡Hola! Soy un chatbot de información de la empresa. Puedes preguntarme sobre las terminales y te proporcionaré el número de teléfono y el correo de facturación, o hacerme preguntas frecuentes.")
    pairs = [
        [
            r"(.*)",
            [lambda matches: str(matches)]  # Simplemente devuelve la entrada del usuario como cadena
        ]
    ]
    chat = MiChat(pairs, reflections)
    chat.converse()

# Ejecución del chatbot
if __name__ == "__main__":
    chatbot()