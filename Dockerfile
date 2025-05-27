# Χρήση της τελευταίας έκδοσης Python ως βάση
FROM python:3.9

# Ορισμός του working directory μέσα στο container
WORKDIR /app

# Αντιγραφή όλων των αρχείων του project στο container
COPY . /app

# Εγκατάσταση των απαραίτητων πακέτων
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir scanpy

# Άνοιγμα της πόρτας 8501 για την Streamlit εφαρμογή
EXPOSE 8501

# Εκκίνηση της εφαρμογής όταν τρέχει το container
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
