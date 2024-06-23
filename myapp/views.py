import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import numpy as np
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

class PredictView(APIView):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = joblib.load('medical_condition_model.pkl')
        self.label_encoders = joblib.load('label_encoders.pkl')
        self.scaler = joblib.load('scaler.pkl')
        self.target_classes = [
            'Hypertension', 'Diabetes Mellitus', 'Asthma', 'Diabetes',
            'Chronic Obstructive Pulmonary Disease (COPD)', 'Coronary Artery Disease',
            'Heart Failure', 'Stroke', 'Chronic Kidney Disease', 'Osteoarthritis',
            'Rheumatoid Arthritis', 'Depression', 'Anxiety Disorder', 'Alzheimer\'s Disease',
            'Parkinson\'s Disease', 'Hyperlipidemia', 'Hypothyroidism', 'Hyperthyroidism',
            'Psoriasis', 'Multiple Sclerosis', 'Lupus', 'Inflammatory Bowel Disease',
            'Gastroesophageal Reflux Disease (GERD)', 'Hepatitis', 'HIV/AIDS',
            'Tuberculosis', 'Malaria', 'COVID-19', 'Cancer', 'Migraine', 'Epilepsy'
        ]
        self.le_target = LabelEncoder()
        self.le_target.fit(self.target_classes)

    def preprocess_input(self, data):
        df = pd.DataFrame(data)
        for column in ['Gender', 'Blood Type', 'Insurance Provider', 'Admission Type', 'Medication']:
            df[column] = df[column].map(
                lambda s: self.label_encoders[column].transform([s])[0] if s in self.label_encoders[column].classes_ else -1
            )
        df = self.scaler.transform(df)
        return df

    def post(self, request, *args, **kwargs):
        data = request.data
        if not data:
            return Response({'error': 'No data provided'}, status=status.HTTP_400_BAD_REQUEST)
        
        preprocessed_data = self.preprocess_input(data)
        prediction = self.model.predict(preprocessed_data)
        predicted_condition = self.le_target.inverse_transform(prediction)
        
        return Response({'predicted_condition': predicted_condition[0]})
