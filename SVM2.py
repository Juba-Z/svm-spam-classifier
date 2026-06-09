import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import joblib

df = pd.read_csv('emails.csv')

print("First 5 rows:")
print(df.head())

print("Null values:")
print(df.isna().sum())

df = df.dropna()  
df.drop_duplicates(inplace=True)      

X = df['text'].str.lower().str.strip()
y = df['spam']


tfidf = TfidfVectorizer(
    stop_words='english',
    max_features=5000,
    ngram_range=(1, 2),
    max_df=0.95,
    min_df=2
)
X_tfidf = tfidf.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

svm_model = LinearSVC(max_iter=2000)
svm_model.fit(X_train, y_train)


y_pred = svm_model.predict(X_test)
acc_svm = accuracy_score(y_test, y_pred)

print("\n SVM Results")
print(classification_report(y_test, y_pred))
print(f"SVM Accuracy: {acc_svm * 100:.2f}%")


def predict_message_svm(message):
    message_clean = message.lower().strip()
    message_tfidf = tfidf.transform([message_clean])
    prediction = svm_model.predict(message_tfidf)[0]

    score = svm_model.decision_function(message_tfidf)[0]
    confidence_str = f" (Score: {score:.4f})"

    label = "Spam" if prediction == 1 else "Not Spam"
    return f"Result: {label}{confidence_str}"


#testing
test_msgs = [
    #spam
    "Congradulations! You have been selected to recieve a free iPhone 15. Claim now!",
    "URGENT: Yur bank account has been suspendded. Verify immedately at: secure-login.xyz",
    "You won a $500 gift card!! Click hear to reddem your prize before it expires.",
    "Loose weight fast with this amazng pill doctors dont want u to know about!",

    # ham
    "Hey, are you comming to the study group tmrw at 3pm? let me know",
    "Can you send me the assignmnt file when you get a chance? thanks",
    "Just a reminder that the metting is moved to Wednessday at 10am",
    "I finished the projet report, ill share it with you after lunch",
]

print("\n Testing Samples -> ")
for i, msg in enumerate(test_msgs, 1):
    print(f"Sample {i}: {predict_message_svm(msg)}")


joblib.dump(svm_model, 'svm_spam_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
print("\n Model Saved Successfully!")