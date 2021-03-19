# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 09:49:40 2020

@author: Mehdi
"""

import numpy as np

a1=np.nanmean([table_21.loc['A'].accuracy,table_22.loc['A'].accuracy,table_23.loc['A'].accuracy,table_24.loc['A'].accuracy,
           table_25.loc['A'].accuracy,table_26.loc['A'].accuracy,table_27.loc['A'].accuracy,table_28.loc['A'].accuracy,
           table_29.loc['A'].accuracy,table_210.loc['A'].accuracy])


a2=np.nanmean([table_21.loc['A'].f1_score,table_22.loc['A'].f1_score,table_23.loc['A'].f1_score,table_24.loc['A'].f1_score,
           table_25.loc['A'].f1_score,table_26.loc['A'].f1_score,table_27.loc['A'].f1_score,table_28.loc['A'].f1_score,
           table_29.loc['A'].f1_score,table_210.loc['A'].f1_score])


a3=np.nanmean([table_21.loc['A'][2],table_22.loc['A'][2],table_23.loc['A'][2],table_24.loc['A'][2],
           table_25.loc['A'][2],table_26.loc['A'][2],table_27.loc['A'][2],table_28.loc['A'][2],
           table_29.loc['A'][2],table_210.loc['A'][2]])


a4=np.nanmean([table_21.loc['A'][3],table_22.loc['A'][3],table_23.loc['A'][3],table_24.loc['A'][3],
           table_25.loc['A'][3],table_26.loc['A'][3],table_27.loc['A'][3],table_28.loc['A'][3],
           table_29.loc['A'][3],table_210.loc['A'][3]])

a5=np.nanmean([table_21.loc['A'][4],table_22.loc['A'][4],table_23.loc['A'][4],table_24.loc['A'][4],
           table_25.loc['A'][4],table_26.loc['A'][4],table_27.loc['A'][4],table_28.loc['A'][4],
           table_29.loc['A'][4],table_210.loc['A'][4]])

a6=np.nanmean([table_21.loc['A'][5],table_22.loc['A'][5],table_23.loc['A'][5],table_24.loc['A'][5],
           table_25.loc['A'][5],table_26.loc['A'][5],table_27.loc['A'][5],table_28.loc['A'][5],
           table_29.loc['A'][5],table_210.loc['A'][5]])

a7=np.nanmean([table_21.loc['B'].accuracy,table_22.loc['B'].accuracy,table_23.loc['B'].accuracy,table_24.loc['B'].accuracy,
           table_25.loc['B'].accuracy,table_26.loc['B'].accuracy,table_27.loc['B'].accuracy,table_28.loc['B'].accuracy,
           table_29.loc['B'].accuracy,table_210.loc['B'].accuracy])


a8=np.nanmean([table_21.loc['B'].f1_score,table_22.loc['B'].f1_score,table_23.loc['B'].f1_score,table_24.loc['B'].f1_score,
           table_25.loc['B'].f1_score,table_26.loc['B'].f1_score,table_27.loc['B'].f1_score,table_28.loc['B'].f1_score,
           table_29.loc['B'].f1_score,table_210.loc['B'].f1_score])


a9=np.nanmean([table_21.loc['B'][2],table_22.loc['B'][2],table_23.loc['B'][2],table_24.loc['B'][2],
           table_25.loc['B'][2],table_26.loc['B'][2],table_27.loc['B'][2],table_28.loc['B'][2],
           table_29.loc['B'][2],table_210.loc['B'][2]])


a10=np.nanmean([table_21.loc['B'][3],table_22.loc['B'][3],table_23.loc['B'][3],table_24.loc['B'][3],
           table_25.loc['B'][3],table_26.loc['B'][3],table_27.loc['B'][3],table_28.loc['B'][3],
           table_29.loc['B'][3],table_210.loc['B'][3]])

a11=np.nanmean([table_21.loc['B'][4],table_22.loc['B'][4],table_23.loc['B'][4],table_24.loc['B'][4],
           table_25.loc['B'][4],table_26.loc['B'][4],table_27.loc['B'][4],table_28.loc['B'][4],
           table_29.loc['B'][4],table_210.loc['B'][4]])

a12=np.nanmean([table_21.loc['B'][5],table_22.loc['B'][5],table_23.loc['B'][5],table_24.loc['B'][5],
           table_25.loc['B'][5],table_26.loc['B'][5],table_27.loc['B'][5],table_28.loc['B'][5],
           table_29.loc['B'][5],table_210.loc['B'][5]])


a13=np.nanmean([table_21.loc['C'].accuracy,table_22.loc['C'].accuracy,table_23.loc['C'].accuracy,table_24.loc['C'].accuracy,
           table_25.loc['C'].accuracy,table_26.loc['C'].accuracy,table_27.loc['C'].accuracy,table_28.loc['C'].accuracy,
           table_29.loc['C'].accuracy,table_210.loc['C'].accuracy])


a14=np.nanmean([table_21.loc['C'].f1_score,table_22.loc['C'].f1_score,table_23.loc['C'].f1_score,table_24.loc['C'].f1_score,
           table_25.loc['C'].f1_score,table_26.loc['C'].f1_score,table_27.loc['C'].f1_score,table_28.loc['C'].f1_score,
           table_29.loc['C'].f1_score,table_210.loc['C'].f1_score])


a15=np.nanmean([table_21.loc['C'][2],table_22.loc['C'][2],table_23.loc['C'][2],table_24.loc['C'][2],
           table_25.loc['C'][2],table_26.loc['C'][2],table_27.loc['C'][2],table_28.loc['C'][2],
           table_29.loc['C'][2],table_210.loc['C'][2]])


a16=np.nanmean([table_21.loc['C'][3],table_22.loc['C'][3],table_23.loc['C'][3],table_24.loc['C'][3],
           table_25.loc['C'][3],table_26.loc['C'][3],table_27.loc['C'][3],table_28.loc['C'][3],
           table_29.loc['C'][3],table_210.loc['C'][3]])

a17=np.nanmean([table_21.loc['C'][4],table_22.loc['C'][4],table_23.loc['C'][4],table_24.loc['C'][4],
           table_25.loc['C'][4],table_26.loc['C'][4],table_27.loc['C'][4],table_28.loc['C'][4],
           table_29.loc['C'][4],table_210.loc['C'][4]])

a18=np.nanmean([table_21.loc['C'][5],table_22.loc['C'][5],table_23.loc['C'][5],table_24.loc['C'][5],
           table_25.loc['C'][5],table_26.loc['C'][5],table_27.loc['C'][5],table_28.loc['C'][5],
           table_29.loc['C'][5],table_210.loc['C'][5]])

a19=np.nanmean([table_21.loc['D'].accuracy,table_22.loc['D'].accuracy,table_23.loc['D'].accuracy,table_24.loc['D'].accuracy,
           table_25.loc['D'].accuracy,table_26.loc['D'].accuracy,table_27.loc['D'].accuracy,table_28.loc['D'].accuracy,
           table_29.loc['D'].accuracy,table_210.loc['D'].accuracy])


a20=np.nanmean([table_21.loc['D'].f1_score,table_22.loc['D'].f1_score,table_23.loc['D'].f1_score,table_24.loc['D'].f1_score,
           table_25.loc['D'].f1_score,table_26.loc['D'].f1_score,table_27.loc['D'].f1_score,table_28.loc['D'].f1_score,
           table_29.loc['D'].f1_score,table_210.loc['D'].f1_score])


a21=np.nanmean([table_21.loc['D'][2],table_22.loc['D'][2],table_23.loc['D'][2],table_24.loc['D'][2],
           table_25.loc['D'][2],table_26.loc['D'][2],table_27.loc['D'][2],table_28.loc['D'][2],
           table_29.loc['D'][2],table_210.loc['D'][2]])


a22=np.nanmean([table_21.loc['D'][3],table_22.loc['D'][3],table_23.loc['D'][3],table_24.loc['D'][3],
           table_25.loc['D'][3],table_26.loc['D'][3],table_27.loc['D'][3],table_28.loc['D'][3],
           table_29.loc['D'][3],table_210.loc['D'][3]])

a23=np.nanmean([table_21.loc['D'][4],table_22.loc['D'][4],table_23.loc['D'][4],table_24.loc['D'][4],
           table_25.loc['D'][4],table_26.loc['D'][4],table_27.loc['D'][4],table_28.loc['D'][4],
           table_29.loc['D'][4],table_210.loc['D'][4]])

a24=np.nanmean([table_21.loc['D'][5],table_22.loc['D'][5],table_23.loc['D'][5],table_24.loc['D'][5],
           table_25.loc['D'][5],table_26.loc['D'][5],table_27.loc['D'][5],table_28.loc['D'][5],
           table_29.loc['D'][5],table_210.loc['D'][5]])


a25=np.nanmean([table_21.loc['E'].accuracy,table_22.loc['E'].accuracy,table_23.loc['E'].accuracy,table_24.loc['E'].accuracy,
           table_25.loc['E'].accuracy,table_26.loc['E'].accuracy,table_27.loc['E'].accuracy,table_28.loc['E'].accuracy,
           table_29.loc['E'].accuracy,table_210.loc['E'].accuracy])


a26=np.nanmean([table_21.loc['E'].f1_score,table_22.loc['E'].f1_score,table_23.loc['E'].f1_score,table_24.loc['E'].f1_score,
           table_25.loc['E'].f1_score,table_26.loc['E'].f1_score,table_27.loc['E'].f1_score,table_28.loc['E'].f1_score,
           table_29.loc['E'].f1_score,table_210.loc['E'].f1_score])


a27=np.nanmean([table_21.loc['E'][2],table_22.loc['E'][2],table_23.loc['E'][2],table_24.loc['E'][2],
           table_25.loc['E'][2],table_26.loc['E'][2],table_27.loc['E'][2],table_28.loc['E'][2],
           table_29.loc['E'][2],table_210.loc['E'][2]])


a28=np.nanmean([table_21.loc['E'][3],table_22.loc['E'][3],table_23.loc['E'][3],table_24.loc['E'][3],
           table_25.loc['E'][3],table_26.loc['E'][3],table_27.loc['E'][3],table_28.loc['E'][3],
           table_29.loc['E'][3],table_210.loc['E'][3]])

a29=np.nanmean([table_21.loc['E'][4],table_22.loc['E'][4],table_23.loc['E'][4],table_24.loc['E'][4],
           table_25.loc['E'][4],table_26.loc['E'][4],table_27.loc['E'][4],table_28.loc['E'][4],
           table_29.loc['E'][4],table_210.loc['E'][4]])

a30=np.nanmean([table_21.loc['E'][5],table_22.loc['E'][5],table_23.loc['E'][5],table_24.loc['E'][5],
           table_25.loc['E'][5],table_26.loc['E'][5],table_27.loc['E'][5],table_28.loc['E'][5],
           table_29.loc['E'][5],table_210.loc['E'][5]])


a31=np.nanmean([table_21.loc['F'].accuracy,table_22.loc['F'].accuracy,table_23.loc['F'].accuracy,table_24.loc['F'].accuracy,
           table_25.loc['F'].accuracy,table_26.loc['F'].accuracy,table_27.loc['F'].accuracy,table_28.loc['F'].accuracy,
           table_29.loc['F'].accuracy,table_210.loc['F'].accuracy])


a32=np.nanmean([table_21.loc['F'].f1_score,table_22.loc['F'].f1_score,table_23.loc['F'].f1_score,table_24.loc['F'].f1_score,
           table_25.loc['F'].f1_score,table_26.loc['F'].f1_score,table_27.loc['F'].f1_score,table_28.loc['F'].f1_score,
           table_29.loc['F'].f1_score,table_210.loc['F'].f1_score])


a33=np.nanmean([table_21.loc['F'][2],table_22.loc['F'][2],table_23.loc['F'][2],table_24.loc['F'][2],
           table_25.loc['F'][2],table_26.loc['F'][2],table_27.loc['F'][2],table_28.loc['F'][2],
           table_29.loc['F'][2],table_210.loc['F'][2]])


a34=np.nanmean([table_21.loc['F'][3],table_22.loc['F'][3],table_23.loc['F'][3],table_24.loc['F'][3],
           table_25.loc['F'][3],table_26.loc['F'][3],table_27.loc['F'][3],table_28.loc['F'][3],
           table_29.loc['F'][3],table_210.loc['F'][3]])

a35=np.nanmean([table_21.loc['F'][4],table_22.loc['F'][4],table_23.loc['F'][4],table_24.loc['F'][4],
           table_25.loc['F'][4],table_26.loc['F'][4],table_27.loc['F'][4],table_28.loc['F'][4],
           table_29.loc['F'][4],table_210.loc['F'][4]])

a36=np.nanmean([table_21.loc['F'][5],table_22.loc['F'][5],table_23.loc['F'][5],table_24.loc['F'][5],
           table_25.loc['F'][5],table_26.loc['F'][5],table_27.loc['F'][5],table_28.loc['F'][5],
           table_29.loc['F'][5],table_210.loc['F'][5]])

a37=np.nanmean([table_21.loc['G'].accuracy,table_22.loc['G'].accuracy,table_23.loc['G'].accuracy,table_24.loc['G'].accuracy,
           table_25.loc['G'].accuracy,table_26.loc['G'].accuracy,table_27.loc['G'].accuracy,table_28.loc['G'].accuracy,
           table_29.loc['G'].accuracy,table_210.loc['G'].accuracy])


a38=np.nanmean([table_21.loc['G'].f1_score,table_22.loc['G'].f1_score,table_23.loc['G'].f1_score,table_24.loc['G'].f1_score,
           table_25.loc['G'].f1_score,table_26.loc['G'].f1_score,table_27.loc['G'].f1_score,table_28.loc['G'].f1_score,
           table_29.loc['G'].f1_score,table_210.loc['G'].f1_score])


a39=np.nanmean([table_21.loc['G'][2],table_22.loc['G'][2],table_23.loc['G'][2],table_24.loc['G'][2],
           table_25.loc['G'][2],table_26.loc['G'][2],table_27.loc['G'][2],table_28.loc['G'][2],
           table_29.loc['G'][2],table_210.loc['G'][2]])


a40=np.nanmean([table_21.loc['G'][3],table_22.loc['G'][3],table_23.loc['G'][3],table_24.loc['G'][3],
           table_25.loc['G'][3],table_26.loc['G'][3],table_27.loc['G'][3],table_28.loc['G'][3],
           table_29.loc['G'][3],table_210.loc['G'][3]])

a41=np.nanmean([table_21.loc['G'][4],table_22.loc['G'][4],table_23.loc['G'][4],table_24.loc['G'][4],
           table_25.loc['G'][4],table_26.loc['G'][4],table_27.loc['G'][4],table_28.loc['G'][4],
           table_29.loc['G'][4],table_210.loc['G'][4]])

a42=np.nanmean([table_21.loc['G'][5],table_22.loc['G'][5],table_23.loc['G'][5],table_24.loc['G'][5],
           table_25.loc['G'][5],table_26.loc['G'][5],table_27.loc['G'][5],table_28.loc['G'][5],
           table_29.loc['G'][5],table_210.loc['G'][5]])


a43=np.nanmean([table_21.loc['H'].accuracy,table_22.loc['H'].accuracy,table_23.loc['H'].accuracy,table_24.loc['H'].accuracy,
           table_25.loc['H'].accuracy,table_26.loc['H'].accuracy,table_27.loc['H'].accuracy,table_28.loc['H'].accuracy,
           table_29.loc['H'].accuracy,table_210.loc['H'].accuracy])


a44=np.nanmean([table_21.loc['H'].f1_score,table_22.loc['H'].f1_score,table_23.loc['H'].f1_score,table_24.loc['H'].f1_score,
           table_25.loc['H'].f1_score,table_26.loc['H'].f1_score,table_27.loc['H'].f1_score,table_28.loc['H'].f1_score,
           table_29.loc['H'].f1_score,table_210.loc['H'].f1_score])


a45=np.nanmean([table_21.loc['H'][2],table_22.loc['H'][2],table_23.loc['H'][2],table_24.loc['H'][2],
           table_25.loc['H'][2],table_26.loc['H'][2],table_27.loc['H'][2],table_28.loc['H'][2],
           table_29.loc['H'][2],table_210.loc['H'][2]])


a46=np.nanmean([table_21.loc['H'][3],table_22.loc['H'][3],table_23.loc['H'][3],table_24.loc['H'][3],
           table_25.loc['H'][3],table_26.loc['H'][3],table_27.loc['H'][3],table_28.loc['H'][3],
           table_29.loc['H'][3],table_210.loc['H'][3]])

a47=np.nanmean([table_21.loc['H'][4],table_22.loc['H'][4],table_23.loc['H'][4],table_24.loc['H'][4],
           table_25.loc['H'][4],table_26.loc['H'][4],table_27.loc['H'][4],table_28.loc['H'][4],
           table_29.loc['H'][4],table_210.loc['H'][4]])

a48=np.nanmean([table_21.loc['H'][5],table_22.loc['H'][5],table_23.loc['H'][5],table_24.loc['H'][5],
           table_25.loc['H'][5],table_26.loc['H'][5],table_27.loc['H'][5],table_28.loc['H'][5],
           table_29.loc['H'][5],table_210.loc['H'][5]])


a49=np.nanmean([table_21.loc['I'].accuracy,table_22.loc['I'].accuracy,table_23.loc['I'].accuracy,table_24.loc['I'].accuracy,
           table_25.loc['I'].accuracy,table_26.loc['I'].accuracy,table_27.loc['I'].accuracy,table_28.loc['I'].accuracy,
           table_29.loc['I'].accuracy,table_210.loc['I'].accuracy])


a50=np.nanmean([table_21.loc['I'].f1_score,table_22.loc['I'].f1_score,table_23.loc['I'].f1_score,table_24.loc['I'].f1_score,
           table_25.loc['I'].f1_score,table_26.loc['I'].f1_score,table_27.loc['I'].f1_score,table_28.loc['I'].f1_score,
           table_29.loc['I'].f1_score,table_210.loc['I'].f1_score])


a51=np.nanmean([table_21.loc['I'][2],table_22.loc['I'][2],table_23.loc['I'][2],table_24.loc['I'][2],
           table_25.loc['I'][2],table_26.loc['I'][2],table_27.loc['I'][2],table_28.loc['I'][2],
           table_29.loc['I'][2],table_210.loc['I'][2]])


a52=np.nanmean([table_21.loc['I'][3],table_22.loc['I'][3],table_23.loc['I'][3],table_24.loc['I'][3],
           table_25.loc['I'][3],table_26.loc['I'][3],table_27.loc['I'][3],table_28.loc['I'][3],
           table_29.loc['I'][3],table_210.loc['I'][3]])

a53=np.nanmean([table_21.loc['I'][4],table_22.loc['I'][4],table_23.loc['I'][4],table_24.loc['I'][4],
           table_25.loc['I'][4],table_26.loc['I'][4],table_27.loc['I'][4],table_28.loc['I'][4],
           table_29.loc['I'][4],table_210.loc['I'][4]])

a54=np.nanmean([table_21.loc['I'][5],table_22.loc['I'][5],table_23.loc['I'][5],table_24.loc['I'][5],
           table_25.loc['I'][5],table_26.loc['I'][5],table_27.loc['I'][5],table_28.loc['I'][5],
           table_29.loc['I'][5],table_210.loc['I'][5]])


a55=np.nanmean([table_21.loc['J'].accuracy,table_22.loc['J'].accuracy,table_23.loc['J'].accuracy,table_24.loc['J'].accuracy,
           table_25.loc['J'].accuracy,table_26.loc['J'].accuracy,table_27.loc['J'].accuracy,table_28.loc['J'].accuracy,
           table_29.loc['J'].accuracy,table_210.loc['J'].accuracy])


a56=np.nanmean([table_21.loc['J'].f1_score,table_22.loc['J'].f1_score,table_23.loc['J'].f1_score,table_24.loc['J'].f1_score,
           table_25.loc['J'].f1_score,table_26.loc['J'].f1_score,table_27.loc['J'].f1_score,table_28.loc['J'].f1_score,
           table_29.loc['J'].f1_score,table_210.loc['J'].f1_score])


a57=np.nanmean([table_21.loc['J'][2],table_22.loc['J'][2],table_23.loc['J'][2],table_24.loc['J'][2],
           table_25.loc['J'][2],table_26.loc['J'][2],table_27.loc['J'][2],table_28.loc['J'][2],
           table_29.loc['J'][2],table_210.loc['J'][2]])


a58=np.nanmean([table_21.loc['J'][3],table_22.loc['J'][3],table_23.loc['J'][3],table_24.loc['J'][3],
           table_25.loc['J'][3],table_26.loc['J'][3],table_27.loc['J'][3],table_28.loc['J'][3],
           table_29.loc['J'][3],table_210.loc['J'][3]])

a59=np.nanmean([table_21.loc['J'][4],table_22.loc['J'][4],table_23.loc['J'][4],table_24.loc['J'][4],
           table_25.loc['J'][4],table_26.loc['J'][4],table_27.loc['J'][4],table_28.loc['J'][4],
           table_29.loc['J'][4],table_210.loc['J'][4]])

a60=np.nanmean([table_21.loc['J'][5],table_22.loc['J'][5],table_23.loc['J'][5],table_24.loc['J'][5],
           table_25.loc['J'][5],table_26.loc['J'][5],table_27.loc['J'][5],table_28.loc['J'][5],
           table_29.loc['J'][5],table_210.loc['J'][5]])

a61=np.nanmean([table_21.loc['K'].accuracy,table_22.loc['K'].accuracy,table_23.loc['K'].accuracy,table_24.loc['K'].accuracy,
           table_25.loc['K'].accuracy,table_26.loc['K'].accuracy,table_27.loc['K'].accuracy,table_28.loc['K'].accuracy,
           table_29.loc['K'].accuracy,table_210.loc['K'].accuracy])


a62=np.nanmean([table_21.loc['K'].f1_score,table_22.loc['K'].f1_score,table_23.loc['K'].f1_score,table_24.loc['K'].f1_score,
           table_25.loc['K'].f1_score,table_26.loc['K'].f1_score,table_27.loc['K'].f1_score,table_28.loc['K'].f1_score,
           table_29.loc['K'].f1_score,table_210.loc['K'].f1_score])


a63=np.nanmean([table_21.loc['K'][2],table_22.loc['K'][2],table_23.loc['K'][2],table_24.loc['K'][2],
           table_25.loc['K'][2],table_26.loc['K'][2],table_27.loc['K'][2],table_28.loc['K'][2],
           table_29.loc['K'][2],table_210.loc['K'][2]])


a64=np.nanmean([table_21.loc['K'][3],table_22.loc['K'][3],table_23.loc['K'][3],table_24.loc['K'][3],
           table_25.loc['K'][3],table_26.loc['K'][3],table_27.loc['K'][3],table_28.loc['K'][3],
           table_29.loc['K'][3],table_210.loc['K'][3]])

a65=np.nanmean([table_21.loc['K'][4],table_22.loc['K'][4],table_23.loc['K'][4],table_24.loc['K'][4],
           table_25.loc['K'][4],table_26.loc['K'][4],table_27.loc['K'][4],table_28.loc['K'][4],
           table_29.loc['K'][4],table_210.loc['K'][4]])

a66=np.nanmean([table_21.loc['K'][5],table_22.loc['K'][5],table_23.loc['K'][5],table_24.loc['K'][5],
           table_25.loc['K'][5],table_26.loc['K'][5],table_27.loc['K'][5],table_28.loc['K'][5],
           table_29.loc['K'][5],table_210.loc['K'][5]])

a67=np.nanmean([table_21.loc['L'].accuracy,table_22.loc['L'].accuracy,table_23.loc['L'].accuracy,table_24.loc['L'].accuracy,
           table_25.loc['L'].accuracy,table_26.loc['L'].accuracy,table_27.loc['L'].accuracy,table_28.loc['L'].accuracy,
           table_29.loc['L'].accuracy,table_210.loc['L'].accuracy])


a68=np.nanmean([table_21.loc['L'].f1_score,table_22.loc['L'].f1_score,table_23.loc['L'].f1_score,table_24.loc['L'].f1_score,
           table_25.loc['L'].f1_score,table_26.loc['L'].f1_score,table_27.loc['L'].f1_score,table_28.loc['L'].f1_score,
           table_29.loc['L'].f1_score,table_210.loc['L'].f1_score])


a69=np.nanmean([table_21.loc['L'][2],table_22.loc['L'][2],table_23.loc['L'][2],table_24.loc['L'][2],
           table_25.loc['L'][2],table_26.loc['L'][2],table_27.loc['L'][2],table_28.loc['L'][2],
           table_29.loc['L'][2],table_210.loc['L'][2]])


a70=np.nanmean([table_21.loc['L'][3],table_22.loc['L'][3],table_23.loc['L'][3],table_24.loc['L'][3],
           table_25.loc['L'][3],table_26.loc['L'][3],table_27.loc['L'][3],table_28.loc['L'][3],
           table_29.loc['L'][3],table_210.loc['L'][3]])

a71=np.nanmean([table_21.loc['L'][4],table_22.loc['L'][4],table_23.loc['L'][4],table_24.loc['L'][4],
           table_25.loc['L'][4],table_26.loc['L'][4],table_27.loc['L'][4],table_28.loc['L'][4],
           table_29.loc['L'][4],table_210.loc['L'][4]])

a72=np.nanmean([table_21.loc['L'][5],table_22.loc['L'][5],table_23.loc['L'][5],table_24.loc['L'][5],
           table_25.loc['L'][5],table_26.loc['L'][5],table_27.loc['L'][5],table_28.loc['L'][5],
           table_29.loc['L'][5],table_210.loc['L'][5]])


a73=np.nanmean([table_21.loc['M'].accuracy,table_22.loc['M'].accuracy,table_23.loc['M'].accuracy,table_24.loc['M'].accuracy,
           table_25.loc['M'].accuracy,table_26.loc['M'].accuracy,table_27.loc['M'].accuracy,table_28.loc['M'].accuracy,
           table_29.loc['M'].accuracy,table_210.loc['M'].accuracy])


a74=np.nanmean([table_21.loc['M'].f1_score,table_22.loc['M'].f1_score,table_23.loc['M'].f1_score,table_24.loc['M'].f1_score,
           table_25.loc['M'].f1_score,table_26.loc['M'].f1_score,table_27.loc['M'].f1_score,table_28.loc['M'].f1_score,
           table_29.loc['M'].f1_score,table_210.loc['M'].f1_score])


a75=np.nanmean([table_21.loc['M'][2],table_22.loc['M'][2],table_23.loc['M'][2],table_24.loc['M'][2],
           table_25.loc['M'][2],table_26.loc['M'][2],table_27.loc['M'][2],table_28.loc['M'][2],
           table_29.loc['M'][2],table_210.loc['M'][2]])


a76=np.nanmean([table_21.loc['M'][3],table_22.loc['M'][3],table_23.loc['M'][3],table_24.loc['M'][3],
           table_25.loc['M'][3],table_26.loc['M'][3],table_27.loc['M'][3],table_28.loc['M'][3],
           table_29.loc['M'][3],table_210.loc['M'][3]])

a77=np.nanmean([table_21.loc['M'][4],table_22.loc['M'][4],table_23.loc['M'][4],table_24.loc['M'][4],
           table_25.loc['M'][4],table_26.loc['M'][4],table_27.loc['M'][4],table_28.loc['M'][4],
           table_29.loc['M'][4],table_210.loc['M'][4]])

a78=np.nanmean([table_21.loc['M'][5],table_22.loc['M'][5],table_23.loc['M'][5],table_24.loc['M'][5],
           table_25.loc['M'][5],table_26.loc['M'][5],table_27.loc['M'][5],table_28.loc['M'][5],
           table_29.loc['M'][5],table_210.loc['M'][5]])


a79=np.nanmean([table_21.loc['.'].accuracy,table_22.loc['.'].accuracy,table_23.loc['.'].accuracy,table_24.loc['.'].accuracy,
           table_25.loc['.'].accuracy,table_26.loc['.'].accuracy,table_27.loc['.'].accuracy,table_28.loc['.'].accuracy,
           table_29.loc['.'].accuracy,table_210.loc['.'].accuracy])


a80=np.nanmean([table_21.loc['.'].f1_score,table_22.loc['.'].f1_score,table_23.loc['.'].f1_score,table_24.loc['.'].f1_score,
           table_25.loc['.'].f1_score,table_26.loc['.'].f1_score,table_27.loc['.'].f1_score,table_28.loc['.'].f1_score,
           table_29.loc['.'].f1_score,table_210.loc['.'].f1_score])


a81=np.nanmean([table_21.loc['.'][2],table_22.loc['.'][2],table_23.loc['.'][2],table_24.loc['.'][2],
           table_25.loc['.'][2],table_26.loc['.'][2],table_27.loc['.'][2],table_28.loc['.'][2],
           table_29.loc['.'][2],table_210.loc['.'][2]])


a82=np.nanmean([table_21.loc['.'][3],table_22.loc['.'][3],table_23.loc['.'][3],table_24.loc['.'][3],
           table_25.loc['.'][3],table_26.loc['.'][3],table_27.loc['.'][3],table_28.loc['.'][3],
           table_29.loc['.'][3],table_210.loc['.'][3]])

a83=np.nanmean([table_21.loc['.'][4],table_22.loc['.'][4],table_23.loc['.'][4],table_24.loc['.'][4],
           table_25.loc['.'][4],table_26.loc['.'][4],table_27.loc['.'][4],table_28.loc['.'][4],
           table_29.loc['.'][4],table_210.loc['.'][4]])

a84=np.nanmean([table_21.loc['.'][5],table_22.loc['.'][5],table_23.loc['.'][5],table_24.loc['.'][5],
           table_25.loc['.'][5],table_26.loc['.'][5],table_27.loc['.'][5],table_28.loc['.'][5],
           table_29.loc['.'][5],table_210.loc['.'][5]])


A=[[a1,a2,a3,round(a4),a5,round(a6)],[a7,a8,a9,round(a10),a11,round(a12)],[a13,a14,a15,round(a16),a17,round(a18)],
    [a19,a20,a21,round(a22),a23,round(a24)]
,[a25,a26,a27,round(a28),a29,round(a30)],[a31,a32,a33,round(a34),a35,round(a36)],
[a37,a38,a39,round(a40),a41,round(a42)],[a43,a44,a45,round(a46),a47,round(a48)],
[a49,a50,a51,round(a52),a53,round(a54)],[a55,a56,a57,round(a58),a59,round(a60)],
[a61,a62,a63,round(a64),a65,round(a66)],[a67,a68,a69,round(a70),a71,round(a72)],
[a73,a74,a75,round(a76),a77,round(a78)],[a79,a80,a81,round(a82),a83,round(a84)]]

vv1=np.mean([v21[0],v22[0],v23[0],v24[0],v25[0],v26[0],v27[0],v28[0],v29[0],v210[0]])
vv2=np.mean([v21[1],v22[1],v23[1],v24[1],v25[1],v26[1],v27[1],v28[1],v29[1],v210[1]])
vv3=np.mean([v21[2],v22[2],v23[2],v24[2],v25[2],v26[2],v27[2],v28[2],v29[2],v210[2]])
vv4=np.mean([v21[3],v22[3],v23[3],v24[3],v25[3],v26[3],v27[3],v28[3],v29[3],v210[3]])
vv5=np.mean([v21[4],v22[4],v23[4],v24[4],v25[4],v26[4],v27[4],v28[4],v29[4],v210[4]])
vv6=np.mean([v21[5],v22[5],v23[5],v24[5],v25[5],v26[5],v27[5],v28[5],v29[5],v210[5]])
table_211= pd.DataFrame(A,columns=['accuracy', 'f1_score', 'accuracy for unknown words',
'number of unknown words','accuracy for known words','number of known words']
,index=['A','B','C','D','E','F','G','H','I','J','K','L','M','.'])

#table_10= pd.DataFrame(A,
#columns=['accuracy', 'f1_score', 'accuracy for unknown words',
#         'number of unknown words','accuracy for known words','number of known words']
#,index=[list(tag2idx.keys())[0], list(tag2idx.keys())[1], list(tag2idx.keys())[2] , list(tag2idx.keys())[3] 
#, list(tag2idx.keys())[4] , list(tag2idx.keys())[5],list(tag2idx.keys())[6],list(tag2idx.keys())[7]
#,list(tag2idx.keys())[8],list(tag2idx.keys())[9],list(tag2idx.keys())[10],list(tag2idx.keys())[11],
#list(tag2idx.keys())[12],list(tag2idx.keys())[13]])

str_pythontex=[float("{0:.2f}".format(list(table_211.loc["A"])[0]*100)),float("{0:.2f}".format(list(table_211.loc["A"])[1]*100)),
float("{0:.2f}".format(list(table_211.loc["A"])[2]*100)),float("{0:.2f}".format(list(table_211.loc["A"])[4]*100)),
round(list(table_211.loc["A"])[3]),round(list(table_211.loc["A"])[5]),
float("{0:.2f}".format(list(table_211.loc["B"])[0]*100)),float("{0:.2f}".format(list(table_211.loc["B"])[1]*100)),
float("{0:.2f}".format(list(table_211.loc["B"])[2]*100)),float("{0:.2f}".format(list(table_211.loc["B"])[4]*100)),
round(list(table_211.loc["B"])[3]),round(list(table_211.loc["B"])[5]),
float("{0:.2f}".format(list(table_211.loc["C"])[0]*100)),float("{0:.2f}".format(list(table_211.loc["C"])[1]*100)),
float("{0:.2f}".format(list(table_211.loc["C"])[2]*100)),float("{0:.2f}".format(list(table_211.loc["C"])[4]*100)),
round(list(table_211.loc["C"])[3]),round(list(table_211.loc["C"])[5]),
float("{0:.2f}".format(list(table_211.loc["D"])[0]*100)),float("{0:.2f}".format(list(table_211.loc["D"])[1]*100)),
float("{0:.2f}".format(list(table_211.loc["D"])[2]*100)),float("{0:.2f}".format(list(table_211.loc["D"])[4]*100)),
round(list(table_211.loc["D"])[3]),round(list(table_211.loc["D"])[5]),
float("{0:.2f}".format(list(table_211.loc["E"])[0]*100)),float("{0:.2f}".format(list(table_211.loc["E"])[1]*100)),
float("{0:.2f}".format(list(table_211.loc["E"])[2]*100)),float("{0:.2f}".format(list(table_211.loc["E"])[4]*100)),
round(list(table_211.loc["E"])[3]),round(list(table_211.loc["E"])[5]),
float("{0:.2f}".format(list(table_211.loc["F"])[0]*100)),float("{0:.2f}".format(list(table_211.loc["F"])[1]*100)),
float("{0:.2f}".format(list(table_211.loc["F"])[2]*100)),float("{0:.2f}".format(list(table_211.loc["F"])[4]*100)),
round(list(table_211.loc["F"])[3]),round(list(table_211.loc["F"])[5]),
float("{0:.2f}".format(list(table_211.loc["G"])[0]*100)),float("{0:.2f}".format(list(table_211.loc["G"])[1]*100)),
float("{0:.2f}".format(list(table_211.loc["G"])[2]*100)),float("{0:.2f}".format(list(table_211.loc["G"])[4]*100)),
round(list(table_211.loc["G"])[3]),round(list(table_211.loc["G"])[5]),
float("{0:.2f}".format(list(table_211.loc["H"])[0]*100)),float("{0:.2f}".format(list(table_211.loc["H"])[1]*100)),
float("{0:.2f}".format(list(table_211.loc["H"])[2]*100)),float("{0:.2f}".format(list(table_211.loc["H"])[4]*100)),
round(list(table_211.loc["H"])[3]),round(list(table_211.loc["H"])[5]),
float("{0:.2f}".format(list(table_211.loc["I"])[0]*100)),float("{0:.2f}".format(list(table_211.loc["I"])[1]*100)),
float("{0:.2f}".format(list(table_211.loc["I"])[2]*100)),float("{0:.2f}".format(list(table_211.loc["I"])[4]*100)),
round(list(table_211.loc["I"])[3]),round(list(table_211.loc["I"])[5]),
float("{0:.2f}".format(list(table_211.loc["J"])[0]*100)),float("{0:.2f}".format(list(table_211.loc["J"])[1]*100)),
float("{0:.2f}".format(list(table_211.loc["J"])[2]*100)),float("{0:.2f}".format(list(table_211.loc["J"])[4]*100)),
round(list(table_211.loc["J"])[3]),round(list(table_211.loc["J"])[5]),
float("{0:.2f}".format(list(table_211.loc["K"])[0]*100)),float("{0:.2f}".format(list(table_211.loc["K"])[1]*100)),
float("{0:.2f}".format(list(table_211.loc["K"])[2]*100)),float("{0:.2f}".format(list(table_211.loc["K"])[4]*100)),
round(list(table_211.loc["K"])[3]),round(list(table_211.loc["K"])[5]),
float("{0:.2f}".format(list(table_211.loc["L"])[0]*100)),float("{0:.2f}".format(list(table_211.loc["L"])[1]*100)),
float("{0:.2f}".format(list(table_211.loc["L"])[2]*100)),float("{0:.2f}".format(list(table_211.loc["L"])[4]*100)),
round(list(table_211.loc["L"])[3]),round(list(table_211.loc["L"])[5]),
float("{0:.2f}".format(list(table_211.loc["M"])[0]*100)),float("{0:.2f}".format(list(table_211.loc["M"])[1]*100)),
float("{0:.2f}".format(list(table_211.loc["M"])[2]*100)),float("{0:.2f}".format(list(table_211.loc["M"])[4]*100)),
round(list(table_211.loc["M"])[3]),round(list(table_211.loc["M"])[5]),
float("{0:.2f}".format(list(table_211.loc["."])[0]*100)),float("{0:.2f}".format(list(table_211.loc["."])[1]*100)),
float("{0:.2f}".format(list(table_211.loc["."])[2]*100)),float("{0:.2f}".format(list(table_211.loc["."])[4]*100)),
round(list(table_211.loc["."])[3]),round(list(table_211.loc["."])[5]),float("{0:.2f}".format(vv1))
,float("{0:.2f}".format(vv2))
,float("{0:.2f}".format(vv3))
,float("{0:.2f}".format(vv4)),round(vv5)
,float("{0:.2f}".format(vv6))
]


L=[]
for x in str_pythontex:
    if math.isnan(x):
        L.append('NULL')
    else:
        L.append(str(x))

L1=[]
i=0
for x in L:
    i=i+1
    if i!=5 and i!=6 and x!="NULL":
        L1.append(x+" \%")
    elif x=="NULL":
        L1.append(x)
    elif i==5:
        L1.append(x)
    else:
        L1.append(x)
        i=0

L1[-1]=L1[-1]+" \%"

