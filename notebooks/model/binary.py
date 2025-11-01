from ultralytics import YOLO
import cv2
import numpy as np

class PlantDiseaseDetector:
    def __init__(self, model_path):
        """Initialise le détecteur de maladies"""
        self.model = YOLO(model_path)
        self.conf_threshold = 0.15  # Seuil optimisé

    def detect(self, image_path, save_path=None):
        """
        Détecte les maladies sur une image

        Args:
            image_path: Chemin vers l'image
            save_path: Chemin pour sauvegarder le résultat

        Returns:
            dict: Résultats de détection
        """
        # Prédiction
        results = self.model.predict(
            source=image_path,
            conf=self.conf_threshold,
            iou=0.45,
            imgsz=640,
            device=0,
            save=save_path is not None,
            save_txt=False,
            show_labels=True,
            show_conf=True
        )

        # Analyse des résultats
        detections = {
            'healthy_count': 0,
            'disease_count': 0,
            'diseases': [],
            'status': 'healthy'
        }

        for r in results:
            for box in r.boxes:
                class_id = int(box.cls)
                class_name = r.names[class_id]
                confidence = float(box.conf)
                bbox = box.xyxy[0].cpu().numpy()

                if class_name == 'disease':
                    detections['disease_count'] += 1
                    detections['diseases'].append({
                        'confidence': confidence,
                        'bbox': bbox.tolist(),
                        'severity': self._assess_severity(confidence)
                    })
                else:
                    detections['healthy_count'] += 1

        # Déterminer le statut global
        if detections['disease_count'] > 0:
            detections['status'] = 'infected'
            avg_conf = np.mean([d['confidence'] for d in detections['diseases']])
            detections['alert_level'] = self._get_alert_level(avg_conf)

        return detections

    def _assess_severity(self, confidence):
        """Évalue la sévérité de la maladie"""
        if confidence >= 0.70:
            return 'high'
        elif confidence >= 0.40:
            return 'medium'
        else:
            return 'low'

    def _get_alert_level(self, avg_confidence):
        """Détermine le niveau d'alerte"""
        if avg_confidence >= 0.60:
            return '🔴 CRITIQUE'
        elif avg_confidence >= 0.30:
            return '🟡 MODÉRÉ'
        else:
            return '🟢 LÉGER'

# Utilisation
detector = PlantDiseaseDetector(
    '/content/runs/detect/PlantDoc-Improved-v2/weights/best.pt'
)

# Test sur une image
result = detector.detect('test_image.jpg', save_path='/content/result.jpg')

print(f"\n{'='*50}")
print(f"📊 RAPPORT DE DIAGNOSTIC")
print(f"{'='*50}")
print(f"🌿 Feuilles saines : {result['healthy_count']}")
print(f"⚠️  Maladies détectées : {result['disease_count']}")
print(f"📈 Statut : {result['status'].upper()}")

if result['disease_count'] > 0:
    print(f"🚨 Niveau d'alerte : {result['alert_level']}")
    print(f"\n📋 Détails des maladies :")
    for i, disease in enumerate(result['diseases'], 1):
        print(f"  {i}. Confiance: {disease['confidence']:.1%} | Sévérité: {disease['severity']}")
