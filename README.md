<h1>Classificação de E-mails como Spam ou Não Spam</h1>
<p>Este repositório contém exemplos de código para a classificação de e-mails em spam e não spam usando diferentes algoritmos de aprendizado de máquina. Os exemplos utilizam as bibliotecas <code>scikit-learn</code> para construir e avaliar modelos de classificação.</p>
    <h2>Requisitos</h2>
    <ul>
        <li>Python 3.6 ou superior</li>
        <li>Bibliotecas <code>numpy</code> e <code>scikit-learn</code></li>
    </ul>
    <p>Você pode instalar as bibliotecas necessárias usando pip:</p>
    <pre><code>pip install numpy scikit-learn</code></pre>
    <h2>Exemplos de Código</h2>
    <h3>1. Classificação com Gaussian Naive Bayes</h3>
    <pre><code>from sklearn.naive_bayes import GaussianNB

# Dados de exemplo
X = [[100, 20], [150, 30], [120, 25], [140, 28]]
y = ['Não Spam', 'Spam', 'Não Spam', 'Spam']

# Treinando o modelo
model = GaussianNB()
model.fit(X, y)

# Previsão para um novo e-mail
novo_email = [[130, 22]]
resultado = model.predict(novo_email)
print(f"Previsão para o novo e-mail: {resultado[0]}")</code></pre>
    <h3>2. Classificação com Multinomial Naive Bayes</h3>
    <pre><code>import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Dados de exemplo
emails = [
    "Oferta imperdível! Ganhe 50% de desconto em todos os produtos!",
    "Parabéns! Você ganhou um prêmio de R$ 10.000! Clique aqui para resgatar.",
    "Você recebeu uma nova mensagem de seu amigo João.",
    "Confira as novas ofertas da loja. Não perca!",
    "Reunião de equipe amanhã às 10h. Por favor, confirme sua presença.",
    "Lembrete: pagamento da fatura do seu cartão de crédito vence amanhã.",
]
labels = [1, 1, 0, 0, 0, 0]  # 1 para spam, 0 para não spam

# Transformar os dados em uma matriz de contagem de palavras (bag of words)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

# Dividir os dados em conjunto de treinamento e conjunto de teste
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Treinar o modelo
model = MultinomialNB()
model.fit(X_train, y_train)

# Fazer previsões
predictions = model.predict(X_test)

# Avaliar a precisão do modelo
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)</code></pre>
    <h3>3. Classificação com K-Nearest Neighbors (KNN)</h3>
    <pre><code>import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Dados de exemplo
emails = [
    "Oferta imperdível! Ganhe 50% de desconto em todos os produtos!",
    "Você ganhou um prêmio de R$ 10.000! Clique aqui para resgatar.",
    "Confira as novas ofertas da loja. Não perca!",
    "Reunião de equipe amanhã às 18h. Por favor, confirme sua presença.",
    "Lembrete: pagamento da fatura do cartão de crédito vence amanhã.",
]
labels = [1, 1, 1, 0, 0]  # 1 para spam, 0 para não spam

# Transformar os dados em uma matriz de contagem de palavras (bag of words)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

# Dividir os dados em conjunto de treinamento e conjunto de teste
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=12)

# Criar e treinar o modelo
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Fazer previsões
predictions = model.predict(X_test)

# Avaliar a precisão do modelo
accuracy = accuracy_score(y_test, predictions)
print("Acurácia:", accuracy)</code></pre>
    <h3>4. Classificação com Support Vector Machine (SVM)</h3>
    <pre><code>import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Dados de exemplo
emails = [
    "Oferta imperdível! Ganhe 50% de desconto em todos os produtos!",
    "Parabéns! Você ganhou um prêmio de R$ 10.000! Clique aqui para resgatar.",
    "Você recebeu uma nova mensagem de seu amigo João.",
    "Confira as novas ofertas da loja. Não perca!",
    "Reunião de equipe amanhã às 10h. Por favor, confirme sua presença.",
    "Lembrete: pagamento da fatura do seu cartão de crédito vence amanhã.",
]
labels = [1, 1, 0, 0, 0, 0]  # 1 para spam, 0 para não spam

# Transformar os dados em uma matriz de contagem de palavras (bag of words)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

# Dividir os dados em conjunto de treinamento e conjunto de teste
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Criar e treinar o modelo
modelo = SVC(kernel='linear')
modelo.fit(X_train, y_train)

# Fazer previsões
predictions = model.predict(X_test)

# Avaliar a precisão do modelo
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)</code></pre>
    <h3>5. Classificação com Decision Tree</h3>
    <pre><code>import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Dados de exemplo
emails = [
    "Oferta imperdível! Ganhe 50% de desconto em todos os produtos!",
    "Parabéns! Você ganhou um prêmio de R$ 10.000! Clique aqui para resgatar.",
    "Você recebeu uma nova mensagem de seu amigo João.",
    "Confira as novas ofertas da loja. Não perca!",
    "Reunião de equipe amanhã às 10h. Por favor, confirme sua presença.",
    "Lembrete: pagamento da fatura do seu cartão de crédito vence amanhã.",
]
labels = [1, 1, 0, 0, 0, 0]  # 1 para spam, 0 para não spam

# Transformar os dados em uma matriz de contagem de palavras (bag of words)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

# Dividir os dados em conjunto de treinamento e conjunto de teste
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Criar e treinar o modelo
modelo = DecisionTreeClassifier()
modelo.fit(X_train, y_train)

# Fazer previsões
predictions = model.predict(X_test)

# Avaliar a precisão do modelo
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)</code></pre>
    <h2>Contribuição</h2>
    <p>Sinta-se à vontade para fazer um fork deste repositório e enviar pull requests. Se você encontrar algum problema, por favor, abra uma issue.</p>
    <h2>Licença</h2>
    <p>Este projeto está licenciado sob os termos da licença MIT.</p>
