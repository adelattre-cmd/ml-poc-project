Mon projet : Prédire si l'action IDACORP (IDA) va sur- ou sous-performer le secteur utilities (XLU) à 20 jours, en utilisant le débit des fleuves de l'Idaho comme
  signal alternatif. Régression supervisée (Ridge, Random Forest, XGBoost).

Le business case: IDACORP produit 50% de son électricité via hydroélectricité. Quand la Snake River est basse (sécheresse), la compagnie achète de l'électricité spot à 
  prix élevé → marges comprimées → le cours sous-performe son secteur 3-6 semaines plus tard. Ce signal physique (débit des rivières) est public mais
  non-suivi par la majorité des investisseurs.

les sources données : 
- USGS NWIS (waterservices.usgs.gov) — débit journalier de 4 fleuves (Columbia, Snake, Willamette, Deschutes) depuis 2000, API publique sans authentification                                                                                     
- Yahoo Finance (yfinance) — prix ajustés quotidiens IDA et XLU depuis 2000  