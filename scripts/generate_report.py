"""Rapport d'une page construit exclusivement à partir de preuves calculées."""
from pathlib import Path
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from pypdf import PdfReader
from src.modeling import MODEL_NAMES
from src.data import ROOT


def generate_report(evidence_path=None, output=None):
    evidence_path = Path(evidence_path or ROOT / 'reports' / 'experiment.json')
    output = Path(output or ROOT / 'reports' / 'one_pager.pdf')
    evidence = json.loads(evidence_path.read_text(encoding='utf-8'))
    selected = next(r for r in evidence['comparison'] if r['name'] == evidence['selected_name'])
    cfg = evidence['config']
    navy = colors.HexColor('#142c50')
    style = ParagraphStyle('body', fontName='Helvetica', fontSize=9, leading=12, textColor=navy, spaceAfter=6)
    heading = ParagraphStyle('heading', parent=style, fontName='Helvetica-Bold', fontSize=10, spaceBefore=6)
    title = ParagraphStyle('title', parent=style, fontName='Helvetica-Bold', fontSize=22, leading=26, spaceAfter=6)
    story = [Paragraph('BankInsight XAI', title),
        Paragraph('Classification explicable pour le marketing bancaire | ENSI - Data Mining 2026/2027', style)]
    def block(name, text):
        story.extend([Paragraph(name, heading), Paragraph(text, style)])
    block('Problème métier et données',
        f"Prioriser les contacts susceptibles de souscrire à un dépôt à terme (y = yes). UCI Bank Marketing : {evidence['dataset_rows']:,} observations, {evidence['positive_rate']:.1%} de positifs. "
        "Décision juste avant l'appel planifié ; duration exclue car connue après l'appel. Jour, mois, canal et rang de campagne supposés connus. Les observations historiques ne mesurent pas un effet causal de campagne.")
    block('Protocole reproductible',
        f"Séparation stratifiée 80/20, graine 42. Entraînement : {cfg['train_rows']:,} lignes "
        f"({'mode rapide : sous-échantillon stratifié' if cfg['train_limit'] else 'mode complet'}), test indépendant : {cfg['test_rows']:,}. "
        f"Imputation, one-hot et standardisation ajustées dans les plis ; recherche de {cfg['search_budget']} candidats, 3 plis, sélection par Average Precision (AP). "
        "Pondération des classes sauf KNN. Aucune optimisation sur le test ; seuil exploratoire de 0,50. Probabilités non calibrées.")
    if not cfg['train_limit'] and (ROOT/'reports/quick_experiment.json').exists():
        quick=json.loads((ROOT/'reports/quick_experiment.json').read_text(encoding='utf-8'))
        q=next(r for r in quick['comparison'] if r['name']==quick['selected_name'])
        story.append(Paragraph(f"Mode rapide (2 500 lignes) : {MODEL_NAMES[q['name']]}, AP test {q['metrics']['average_precision']:.3f}. "
            f"Mode complet : {MODEL_NAMES[selected['name']]}, AP test {selected['metrics']['average_precision']:.3f}. "
            "Même holdout déjà consulté : comparaison descriptive, aucun réglage guidé par le test.",style))
    story.append(Paragraph('Comparaison réelle - modèle recommandé : ' + MODEL_NAMES[selected['name']], heading))
    rows = [['Modèle', 'AP CV', 'AP test', 'ROC-AUC', 'Précision', 'Rappel', 'F1']]
    for r in evidence['comparison']:
        m = r['metrics']
        rows.append([MODEL_NAMES[r['name']], f"{r['cv_ap']:.3f}", *[f'{m[k]:.3f}' for k in ['average_precision','roc_auc','precision','recall','f1']]])
    table = Table(rows, colWidths=[144, 49, 49, 57, 57, 49, 49], hAlign='LEFT')
    table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),navy),('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),6),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#eef3f9'),colors.white]),('VALIGN',(0,0),(-1,-1),'MIDDLE')]))
    story.append(table)
    global_result = evidence['global_explanation']
    top = ', '.join(f'{name} ({value:.3f})' for name, value in global_result['ranking'][:3])
    block('Explications SHAP et exemple individuel',
        f"Variables dominantes, moyenne |SHAP| en probabilité : {top}. SHAP permutation marginale, {global_result['sample_size']} clients du test, "
        f"{global_result['background_size']} références d'entraînement et 3 permutations. Erreur d'additivité maximale : {global_result['additivity_error']:.1e}. "
        + evidence['local_explanation']['summary'])
    local = evidence['local_explanation']
    importance=dict(global_result['ranking'])
    sensitive=f"Sur l'échantillon SHAP, moyenne |âge| = {importance.get('age',0):.3f} et |situation familiale| = {importance.get('marital',0):.3f}. "
    agreement = 'indéterminé' if local['top_sign_agreement'] is None else f"{local['top_sign_agreement']:.0%}"
    block('Regard critique : SHAP / LIME et équité',
        f"Même modèle, client et classe positive. Accord de signe sur les principaux facteurs SHAP : {agreement} ; fidélité locale LIME R² = {local['lime']['score']:.3f}. "
        "L'additivité SHAP est vérifiée pour les références échantillonnées ; LIME ajuste un substitut local sur 700 perturbations. "
        "Les voisinages et variables corrélées peuvent produire des divergences. " + sensitive +
        "Comparer les taux de ciblage et rappels par groupe (effectifs dans l'application), sans conclure à une certification d'équité. "
        + ("La faible fidélité LIME limite ici la portée de ses poids. " if local['lime']['score']<.5 else ''))
    logistic = next((r for r in evidence['comparison'] if r['name']=='logistic'), None)
    xgb = next((r for r in evidence['comparison'] if r['name']=='xgboost'), None)
    tradeoff = f"AP CV : logistique {logistic['cv_ap']:.3f}, XGBoost {xgb['cv_ap']:.3f}. " if logistic and xgb else ''
    block('Recommandations et limites', tradeoff +
        "Retenir le classement CV pour la démonstration ; préférer la logistique si sa perte de performance est acceptable face au besoin d'audit des coefficients. "
        "Valider prospectivement, fixer les coûts des faux positifs (sollicitations inutiles) et faux négatifs (occasions manquées), calibrer sur l'entraînement et auditer les groupes avant tout usage. "
        "Données anciennes, dépendances temporelles et clients répétés possibles, séparation aléatoire non prospective, petit budget de recherche et explications échantillonnées limitent la généralisation. Ne pas déduire des actions causales des attributions.")
    story.extend([Spacer(1,6), Paragraph('Source : Moro, S., Rita, P. et Cortez, P. (2014), Bank Marketing, UCI, DOI 10.24432/C5K306, CC BY 4.0. '
        'Résultats traçables : reports/experiment.json. Cahier des charges : PDF officiel, pages 1-4.', ParagraphStyle('source',parent=style,fontSize=7.5,leading=10))])
    output.parent.mkdir(parents=True, exist_ok=True)
    SimpleDocTemplate(str(output), pagesize=A4, rightMargin=24, leftMargin=24, topMargin=25, bottomMargin=24).build(story)
    if len(PdfReader(output).pages) != 1:
        raise RuntimeError("Le rapport dépasse une page : corriger la mise en page.")
    return output


if __name__ == '__main__':
    print(generate_report())
