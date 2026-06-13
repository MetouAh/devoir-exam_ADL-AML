#!/usr/bin/env python3
"""
Script de lancement — Pipeline Hybride DL × ML
Exécute les étapes principales sans interface Jupyter.
Pour une exécution complète avec visualisations, utiliser le notebook Colab.
"""

import os
import sys
import argparse

def check_dependencies():
    """Vérifie que les dépendances sont installées."""
    required = [
        'tensorflow', 'sklearn', 'xgboost', 'lightgbm',
        'shap', 'umap', 'pandas', 'numpy', 'matplotlib'
    ]
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"❌ Dépendances manquantes : {', '.join(missing)}")
        print("   Exécutez : pip install -r requirements.txt")
        sys.exit(1)
    print("✅ Toutes les dépendances sont disponibles")


def run_plantvillage(cfg):
    """Lance le pipeline PlantVillage."""
    import random, numpy as np
    import tensorflow as tf

    SEED = cfg['seed']
    random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)

    print("\n" + "="*60)
    print("DATASET A — PlantVillage")
    print("="*60)
    print("→ Ce pipeline nécessite Kaggle API pour télécharger PlantVillage.")
    print("  Configurer ~/.kaggle/kaggle.json avant de continuer.")
    print("  Ou utiliser le notebook Colab pour une exécution interactive.")

    if cfg.get('dry_run'):
        print("\n[DRY RUN] Pipeline PlantVillage simulé avec succès.")
        return {
            'DL_seul': {'Accuracy': 0.0, 'F1': 0.0},
            'Hybride': {'Accuracy': 0.0, 'F1': 0.0},
        }

    # Pour exécution réelle, importer et lancer le notebook converti
    print("\n⚠️  Exécution complète disponible dans le notebook Colab.")
    return {}


def run_electric(cfg):
    """Lance le pipeline Consommation Électrique."""
    import numpy as np
    import pandas as pd

    SEED = cfg['seed']
    np.random.seed(SEED)

    print("\n" + "="*60)
    print("DATASET B — Consommation Électrique")
    print("="*60)

    if cfg.get('dry_run'):
        print("\n[DRY RUN] Pipeline Électrique simulé avec succès.")
        return {
            'DL_seul': {'Accuracy': 0.0, 'F1': 0.0},
            'Hybride': {'Accuracy': 0.0, 'F1': 0.0},
        }

    print("→ Téléchargement UCI Household Power Consumption...")
    print("⚠️  Exécution complète disponible dans le notebook Colab.")
    return {}


def print_summary(results_pv, results_e):
    """Affiche le tableau de bord synthétique."""
    print("\n" + "="*70)
    print("SYNTHÈSE — PIPELINE HYBRIDE DL × ML")
    print("="*70)

    for ds_name, results in [("PlantVillage", results_pv), ("Électrique", results_e)]:
        if results:
            print(f"\n📊 {ds_name}")
            for method, res in results.items():
                if isinstance(res, dict) and 'Accuracy' in res:
                    print(f"  {method:12s} → Accuracy: {res['Accuracy']:.4f} | F1: {res.get('F1', 0):.4f}")


def main():
    parser = argparse.ArgumentParser(
        description='Pipeline Hybride DL × ML — Lancement',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python run_pipeline.py --dry-run          # Test sans données réelles
  python run_pipeline.py --dataset electric  # Électrique uniquement
  python run_pipeline.py --seed 123          # Graine personnalisée

Note : Pour l'exécution complète avec GPU et visualisations,
       utiliser le notebook Google Colab (recommandé).
        """
    )
    parser.add_argument('--dataset', choices=['plantvillage', 'electric', 'both'],
                        default='both', help='Dataset(s) à traiter (défaut: both)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Graine aléatoire (défaut: 42)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Simulation sans données réelles')
    parser.add_argument('--output-dir', default='./outputs',
                        help='Répertoire de sortie (défaut: ./outputs)')

    args = parser.parse_args()

    print("🚀 Pipeline Hybride DL × ML")
    print(f"   Seed     : {args.seed}")
    print(f"   Dataset  : {args.dataset}")
    print(f"   Dry run  : {args.dry_run}")
    print(f"   Outputs  : {args.output_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    check_dependencies()

    cfg = {
        'seed': args.seed,
        'dry_run': args.dry_run,
        'output_dir': args.output_dir,
        'pv_img_size': 128,
        'pv_batch_size': 32,
        'pv_epochs': 10,
        'pv_n_classes': 10,
        'pv_samples': 5000,
        'elec_window': 24,
        'elec_epochs': 20,
        'elec_batch': 64,
        'pca_components': 50,
        'umap_components': 20,
        'n_splits_cv': 5,
    }

    results_pv, results_e = {}, {}

    if args.dataset in ('plantvillage', 'both'):
        results_pv = run_plantvillage(cfg)

    if args.dataset in ('electric', 'both'):
        results_e = run_electric(cfg)

    print_summary(results_pv, results_e)
    print(f"\n✅ Terminé. Résultats dans : {args.output_dir}/")


if __name__ == '__main__':
    main()
