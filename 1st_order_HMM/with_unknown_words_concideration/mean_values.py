# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 09:49:40 2020

@author: Mehdi
"""

import numpy as np

a1=np.nanmean([table_1.loc['A'].accuracy,table_2.loc['A'].accuracy,table_3.loc['A'].accuracy,table_4.loc['A'].accuracy,
           table_5.loc['A'].accuracy,table_6.loc['A'].accuracy,table_7.loc['A'].accuracy,table_8.loc['A'].accuracy,
           table_9.loc['A'].accuracy,table_10.loc['A'].accuracy])


a2=np.nanmean([table_1.loc['A'].f1_score,table_2.loc['A'].f1_score,table_3.loc['A'].f1_score,table_4.loc['A'].f1_score,
           table_5.loc['A'].f1_score,table_6.loc['A'].f1_score,table_7.loc['A'].f1_score,table_8.loc['A'].f1_score,
           table_9.loc['A'].f1_score,table_10.loc['A'].f1_score])


a3=np.nanmean([table_1.loc['A'][2],table_2.loc['A'][2],table_3.loc['A'][2],table_4.loc['A'][2],
           table_5.loc['A'][2],table_6.loc['A'][2],table_7.loc['A'][2],table_8.loc['A'][2],
           table_9.loc['A'][2],table_10.loc['A'][2]])


a4=np.nanmean([table_1.loc['A'][3],table_2.loc['A'][3],table_3.loc['A'][3],table_4.loc['A'][3],
           table_5.loc['A'][3],table_6.loc['A'][3],table_7.loc['A'][3],table_8.loc['A'][3],
           table_9.loc['A'][3],table_10.loc['A'][3]])

a5=np.nanmean([table_1.loc['A'][4],table_2.loc['A'][4],table_3.loc['A'][4],table_4.loc['A'][4],
           table_5.loc['A'][4],table_6.loc['A'][4],table_7.loc['A'][4],table_8.loc['A'][4],
           table_9.loc['A'][4],table_10.loc['A'][4]])

a6=np.nanmean([table_1.loc['A'][5],table_2.loc['A'][5],table_3.loc['A'][5],table_4.loc['A'][5],
           table_5.loc['A'][5],table_6.loc['A'][5],table_7.loc['A'][5],table_8.loc['A'][5],
           table_9.loc['A'][5],table_10.loc['A'][5]])

a7=np.nanmean([table_1.loc['B'].accuracy,table_2.loc['B'].accuracy,table_3.loc['B'].accuracy,table_4.loc['B'].accuracy,
           table_5.loc['B'].accuracy,table_6.loc['B'].accuracy,table_7.loc['B'].accuracy,table_8.loc['B'].accuracy,
           table_9.loc['B'].accuracy,table_10.loc['B'].accuracy])


a8=np.nanmean([table_1.loc['B'].f1_score,table_2.loc['B'].f1_score,table_3.loc['B'].f1_score,table_4.loc['B'].f1_score,
           table_5.loc['B'].f1_score,table_6.loc['B'].f1_score,table_7.loc['B'].f1_score,table_8.loc['B'].f1_score,
           table_9.loc['B'].f1_score,table_10.loc['B'].f1_score])


a9=np.nanmean([table_1.loc['B'][2],table_2.loc['B'][2],table_3.loc['B'][2],table_4.loc['B'][2],
           table_5.loc['B'][2],table_6.loc['B'][2],table_7.loc['B'][2],table_8.loc['B'][2],
           table_9.loc['B'][2],table_10.loc['B'][2]])


a10=np.nanmean([table_1.loc['B'][3],table_2.loc['B'][3],table_3.loc['B'][3],table_4.loc['B'][3],
           table_5.loc['B'][3],table_6.loc['B'][3],table_7.loc['B'][3],table_8.loc['B'][3],
           table_9.loc['B'][3],table_10.loc['B'][3]])

a11=np.nanmean([table_1.loc['B'][4],table_2.loc['B'][4],table_3.loc['B'][4],table_4.loc['B'][4],
           table_5.loc['B'][4],table_6.loc['B'][4],table_7.loc['B'][4],table_8.loc['B'][4],
           table_9.loc['B'][4],table_10.loc['B'][4]])

a12=np.nanmean([table_1.loc['B'][5],table_2.loc['B'][5],table_3.loc['B'][5],table_4.loc['B'][5],
           table_5.loc['B'][5],table_6.loc['B'][5],table_7.loc['B'][5],table_8.loc['B'][5],
           table_9.loc['B'][5],table_10.loc['B'][5]])


a13=np.nanmean([table_1.loc['C'].accuracy,table_2.loc['C'].accuracy,table_3.loc['C'].accuracy,table_4.loc['C'].accuracy,
           table_5.loc['C'].accuracy,table_6.loc['C'].accuracy,table_7.loc['C'].accuracy,table_8.loc['C'].accuracy,
           table_9.loc['C'].accuracy,table_10.loc['C'].accuracy])


a14=np.nanmean([table_1.loc['C'].f1_score,table_2.loc['C'].f1_score,table_3.loc['C'].f1_score,table_4.loc['C'].f1_score,
           table_5.loc['C'].f1_score,table_6.loc['C'].f1_score,table_7.loc['C'].f1_score,table_8.loc['C'].f1_score,
           table_9.loc['C'].f1_score,table_10.loc['C'].f1_score])


a15=np.nanmean([table_1.loc['C'][2],table_2.loc['C'][2],table_3.loc['C'][2],table_4.loc['C'][2],
           table_5.loc['C'][2],table_6.loc['C'][2],table_7.loc['C'][2],table_8.loc['C'][2],
           table_9.loc['C'][2],table_10.loc['C'][2]])


a16=np.nanmean([table_1.loc['C'][3],table_2.loc['C'][3],table_3.loc['C'][3],table_4.loc['C'][3],
           table_5.loc['C'][3],table_6.loc['C'][3],table_7.loc['C'][3],table_8.loc['C'][3],
           table_9.loc['C'][3],table_10.loc['C'][3]])

a17=np.nanmean([table_1.loc['C'][4],table_2.loc['C'][4],table_3.loc['C'][4],table_4.loc['C'][4],
           table_5.loc['C'][4],table_6.loc['C'][4],table_7.loc['C'][4],table_8.loc['C'][4],
           table_9.loc['C'][4],table_10.loc['C'][4]])

a18=np.nanmean([table_1.loc['C'][5],table_2.loc['C'][5],table_3.loc['C'][5],table_4.loc['C'][5],
           table_5.loc['C'][5],table_6.loc['C'][5],table_7.loc['C'][5],table_8.loc['C'][5],
           table_9.loc['C'][5],table_10.loc['C'][5]])

a19=np.nanmean([table_1.loc['D'].accuracy,table_2.loc['D'].accuracy,table_3.loc['D'].accuracy,table_4.loc['D'].accuracy,
           table_5.loc['D'].accuracy,table_6.loc['D'].accuracy,table_7.loc['D'].accuracy,table_8.loc['D'].accuracy,
           table_9.loc['D'].accuracy,table_10.loc['D'].accuracy])


a20=np.nanmean([table_1.loc['D'].f1_score,table_2.loc['D'].f1_score,table_3.loc['D'].f1_score,table_4.loc['D'].f1_score,
           table_5.loc['D'].f1_score,table_6.loc['D'].f1_score,table_7.loc['D'].f1_score,table_8.loc['D'].f1_score,
           table_9.loc['D'].f1_score,table_10.loc['D'].f1_score])


a21=np.nanmean([table_1.loc['D'][2],table_2.loc['D'][2],table_3.loc['D'][2],table_4.loc['D'][2],
           table_5.loc['D'][2],table_6.loc['D'][2],table_7.loc['D'][2],table_8.loc['D'][2],
           table_9.loc['D'][2],table_10.loc['D'][2]])


a22=np.nanmean([table_1.loc['D'][3],table_2.loc['D'][3],table_3.loc['D'][3],table_4.loc['D'][3],
           table_5.loc['D'][3],table_6.loc['D'][3],table_7.loc['D'][3],table_8.loc['D'][3],
           table_9.loc['D'][3],table_10.loc['D'][3]])

a23=np.nanmean([table_1.loc['D'][4],table_2.loc['D'][4],table_3.loc['D'][4],table_4.loc['D'][4],
           table_5.loc['D'][4],table_6.loc['D'][4],table_7.loc['D'][4],table_8.loc['D'][4],
           table_9.loc['D'][4],table_10.loc['D'][4]])

a24=np.nanmean([table_1.loc['D'][5],table_2.loc['D'][5],table_3.loc['D'][5],table_4.loc['D'][5],
           table_5.loc['D'][5],table_6.loc['D'][5],table_7.loc['D'][5],table_8.loc['D'][5],
           table_9.loc['D'][5],table_10.loc['D'][5]])


a25=np.nanmean([table_1.loc['E'].accuracy,table_2.loc['E'].accuracy,table_3.loc['E'].accuracy,table_4.loc['E'].accuracy,
           table_5.loc['E'].accuracy,table_6.loc['E'].accuracy,table_7.loc['E'].accuracy,table_8.loc['E'].accuracy,
           table_9.loc['E'].accuracy,table_10.loc['E'].accuracy])


a26=np.nanmean([table_1.loc['E'].f1_score,table_2.loc['E'].f1_score,table_3.loc['E'].f1_score,table_4.loc['E'].f1_score,
           table_5.loc['E'].f1_score,table_6.loc['E'].f1_score,table_7.loc['E'].f1_score,table_8.loc['E'].f1_score,
           table_9.loc['E'].f1_score,table_10.loc['E'].f1_score])


a27=np.nanmean([table_1.loc['E'][2],table_2.loc['E'][2],table_3.loc['E'][2],table_4.loc['E'][2],
           table_5.loc['E'][2],table_6.loc['E'][2],table_7.loc['E'][2],table_8.loc['E'][2],
           table_9.loc['E'][2],table_10.loc['E'][2]])


a28=np.nanmean([table_1.loc['E'][3],table_2.loc['E'][3],table_3.loc['E'][3],table_4.loc['E'][3],
           table_5.loc['E'][3],table_6.loc['E'][3],table_7.loc['E'][3],table_8.loc['E'][3],
           table_9.loc['E'][3],table_10.loc['E'][3]])

a29=np.nanmean([table_1.loc['E'][4],table_2.loc['E'][4],table_3.loc['E'][4],table_4.loc['E'][4],
           table_5.loc['E'][4],table_6.loc['E'][4],table_7.loc['E'][4],table_8.loc['E'][4],
           table_9.loc['E'][4],table_10.loc['E'][4]])

a30=np.nanmean([table_1.loc['E'][5],table_2.loc['E'][5],table_3.loc['E'][5],table_4.loc['E'][5],
           table_5.loc['E'][5],table_6.loc['E'][5],table_7.loc['E'][5],table_8.loc['E'][5],
           table_9.loc['E'][5],table_10.loc['E'][5]])


a31=np.nanmean([table_1.loc['F'].accuracy,table_2.loc['F'].accuracy,table_3.loc['F'].accuracy,table_4.loc['F'].accuracy,
           table_5.loc['F'].accuracy,table_6.loc['F'].accuracy,table_7.loc['F'].accuracy,table_8.loc['F'].accuracy,
           table_9.loc['F'].accuracy,table_10.loc['F'].accuracy])


a32=np.nanmean([table_1.loc['F'].f1_score,table_2.loc['F'].f1_score,table_3.loc['F'].f1_score,table_4.loc['F'].f1_score,
           table_5.loc['F'].f1_score,table_6.loc['F'].f1_score,table_7.loc['F'].f1_score,table_8.loc['F'].f1_score,
           table_9.loc['F'].f1_score,table_10.loc['F'].f1_score])


a33=np.nanmean([table_1.loc['F'][2],table_2.loc['F'][2],table_3.loc['F'][2],table_4.loc['F'][2],
           table_5.loc['F'][2],table_6.loc['F'][2],table_7.loc['F'][2],table_8.loc['F'][2],
           table_9.loc['F'][2],table_10.loc['F'][2]])


a34=np.nanmean([table_1.loc['F'][3],table_2.loc['F'][3],table_3.loc['F'][3],table_4.loc['F'][3],
           table_5.loc['F'][3],table_6.loc['F'][3],table_7.loc['F'][3],table_8.loc['F'][3],
           table_9.loc['F'][3],table_10.loc['F'][3]])

a35=np.nanmean([table_1.loc['F'][4],table_2.loc['F'][4],table_3.loc['F'][4],table_4.loc['F'][4],
           table_5.loc['F'][4],table_6.loc['F'][4],table_7.loc['F'][4],table_8.loc['F'][4],
           table_9.loc['F'][4],table_10.loc['F'][4]])

a36=np.nanmean([table_1.loc['F'][5],table_2.loc['F'][5],table_3.loc['F'][5],table_4.loc['F'][5],
           table_5.loc['F'][5],table_6.loc['F'][5],table_7.loc['F'][5],table_8.loc['F'][5],
           table_9.loc['F'][5],table_10.loc['F'][5]])

a37=np.nanmean([table_1.loc['G'].accuracy,table_2.loc['G'].accuracy,table_3.loc['G'].accuracy,table_4.loc['G'].accuracy,
           table_5.loc['G'].accuracy,table_6.loc['G'].accuracy,table_7.loc['G'].accuracy,table_8.loc['G'].accuracy,
           table_9.loc['G'].accuracy,table_10.loc['G'].accuracy])


a38=np.nanmean([table_1.loc['G'].f1_score,table_2.loc['G'].f1_score,table_3.loc['G'].f1_score,table_4.loc['G'].f1_score,
           table_5.loc['G'].f1_score,table_6.loc['G'].f1_score,table_7.loc['G'].f1_score,table_8.loc['G'].f1_score,
           table_9.loc['G'].f1_score,table_10.loc['G'].f1_score])


a39=np.nanmean([table_1.loc['G'][2],table_2.loc['G'][2],table_3.loc['G'][2],table_4.loc['G'][2],
           table_5.loc['G'][2],table_6.loc['G'][2],table_7.loc['G'][2],table_8.loc['G'][2],
           table_9.loc['G'][2],table_10.loc['G'][2]])


a40=np.nanmean([table_1.loc['G'][3],table_2.loc['G'][3],table_3.loc['G'][3],table_4.loc['G'][3],
           table_5.loc['G'][3],table_6.loc['G'][3],table_7.loc['G'][3],table_8.loc['G'][3],
           table_9.loc['G'][3],table_10.loc['G'][3]])

a41=np.nanmean([table_1.loc['G'][4],table_2.loc['G'][4],table_3.loc['G'][4],table_4.loc['G'][4],
           table_5.loc['G'][4],table_6.loc['G'][4],table_7.loc['G'][4],table_8.loc['G'][4],
           table_9.loc['G'][4],table_10.loc['G'][4]])

a42=np.nanmean([table_1.loc['G'][5],table_2.loc['G'][5],table_3.loc['G'][5],table_4.loc['G'][5],
           table_5.loc['G'][5],table_6.loc['G'][5],table_7.loc['G'][5],table_8.loc['G'][5],
           table_9.loc['G'][5],table_10.loc['G'][5]])


a43=np.nanmean([table_1.loc['H'].accuracy,table_2.loc['H'].accuracy,table_3.loc['H'].accuracy,table_4.loc['H'].accuracy,
           table_5.loc['H'].accuracy,table_6.loc['H'].accuracy,table_7.loc['H'].accuracy,table_8.loc['H'].accuracy,
           table_9.loc['H'].accuracy,table_10.loc['H'].accuracy])


a44=np.nanmean([table_1.loc['H'].f1_score,table_2.loc['H'].f1_score,table_3.loc['H'].f1_score,table_4.loc['H'].f1_score,
           table_5.loc['H'].f1_score,table_6.loc['H'].f1_score,table_7.loc['H'].f1_score,table_8.loc['H'].f1_score,
           table_9.loc['H'].f1_score,table_10.loc['H'].f1_score])


a45=np.nanmean([table_1.loc['H'][2],table_2.loc['H'][2],table_3.loc['H'][2],table_4.loc['H'][2],
           table_5.loc['H'][2],table_6.loc['H'][2],table_7.loc['H'][2],table_8.loc['H'][2],
           table_9.loc['H'][2],table_10.loc['H'][2]])


a46=np.nanmean([table_1.loc['H'][3],table_2.loc['H'][3],table_3.loc['H'][3],table_4.loc['H'][3],
           table_5.loc['H'][3],table_6.loc['H'][3],table_7.loc['H'][3],table_8.loc['H'][3],
           table_9.loc['H'][3],table_10.loc['H'][3]])

a47=np.nanmean([table_1.loc['H'][4],table_2.loc['H'][4],table_3.loc['H'][4],table_4.loc['H'][4],
           table_5.loc['H'][4],table_6.loc['H'][4],table_7.loc['H'][4],table_8.loc['H'][4],
           table_9.loc['H'][4],table_10.loc['H'][4]])

a48=np.nanmean([table_1.loc['H'][5],table_2.loc['H'][5],table_3.loc['H'][5],table_4.loc['H'][5],
           table_5.loc['H'][5],table_6.loc['H'][5],table_7.loc['H'][5],table_8.loc['H'][5],
           table_9.loc['H'][5],table_10.loc['H'][5]])


a49=np.nanmean([table_1.loc['I'].accuracy,table_2.loc['I'].accuracy,table_3.loc['I'].accuracy,table_4.loc['I'].accuracy,
           table_5.loc['I'].accuracy,table_6.loc['I'].accuracy,table_7.loc['I'].accuracy,table_8.loc['I'].accuracy,
           table_9.loc['I'].accuracy,table_10.loc['I'].accuracy])


a50=np.nanmean([table_1.loc['I'].f1_score,table_2.loc['I'].f1_score,table_3.loc['I'].f1_score,table_4.loc['I'].f1_score,
           table_5.loc['I'].f1_score,table_6.loc['I'].f1_score,table_7.loc['I'].f1_score,table_8.loc['I'].f1_score,
           table_9.loc['I'].f1_score,table_10.loc['I'].f1_score])


a51=np.nanmean([table_1.loc['I'][2],table_2.loc['I'][2],table_3.loc['I'][2],table_4.loc['I'][2],
           table_5.loc['I'][2],table_6.loc['I'][2],table_7.loc['I'][2],table_8.loc['I'][2],
           table_9.loc['I'][2],table_10.loc['I'][2]])


a52=np.nanmean([table_1.loc['I'][3],table_2.loc['I'][3],table_3.loc['I'][3],table_4.loc['I'][3],
           table_5.loc['I'][3],table_6.loc['I'][3],table_7.loc['I'][3],table_8.loc['I'][3],
           table_9.loc['I'][3],table_10.loc['I'][3]])

a53=np.nanmean([table_1.loc['I'][4],table_2.loc['I'][4],table_3.loc['I'][4],table_4.loc['I'][4],
           table_5.loc['I'][4],table_6.loc['I'][4],table_7.loc['I'][4],table_8.loc['I'][4],
           table_9.loc['I'][4],table_10.loc['I'][4]])

a54=np.nanmean([table_1.loc['I'][5],table_2.loc['I'][5],table_3.loc['I'][5],table_4.loc['I'][5],
           table_5.loc['I'][5],table_6.loc['I'][5],table_7.loc['I'][5],table_8.loc['I'][5],
           table_9.loc['I'][5],table_10.loc['I'][5]])


a55=np.nanmean([table_1.loc['J'].accuracy,table_2.loc['J'].accuracy,table_3.loc['J'].accuracy,table_4.loc['J'].accuracy,
           table_5.loc['J'].accuracy,table_6.loc['J'].accuracy,table_7.loc['J'].accuracy,table_8.loc['J'].accuracy,
           table_9.loc['J'].accuracy,table_10.loc['J'].accuracy])


a56=np.nanmean([table_1.loc['J'].f1_score,table_2.loc['J'].f1_score,table_3.loc['J'].f1_score,table_4.loc['J'].f1_score,
           table_5.loc['J'].f1_score,table_6.loc['J'].f1_score,table_7.loc['J'].f1_score,table_8.loc['J'].f1_score,
           table_9.loc['J'].f1_score,table_10.loc['J'].f1_score])


a57=np.nanmean([table_1.loc['J'][2],table_2.loc['J'][2],table_3.loc['J'][2],table_4.loc['J'][2],
           table_5.loc['J'][2],table_6.loc['J'][2],table_7.loc['J'][2],table_8.loc['J'][2],
           table_9.loc['J'][2],table_10.loc['J'][2]])


a58=np.nanmean([table_1.loc['J'][3],table_2.loc['J'][3],table_3.loc['J'][3],table_4.loc['J'][3],
           table_5.loc['J'][3],table_6.loc['J'][3],table_7.loc['J'][3],table_8.loc['J'][3],
           table_9.loc['J'][3],table_10.loc['J'][3]])

a59=np.nanmean([table_1.loc['J'][4],table_2.loc['J'][4],table_3.loc['J'][4],table_4.loc['J'][4],
           table_5.loc['J'][4],table_6.loc['J'][4],table_7.loc['J'][4],table_8.loc['J'][4],
           table_9.loc['J'][4],table_10.loc['J'][4]])

a60=np.nanmean([table_1.loc['J'][5],table_2.loc['J'][5],table_3.loc['J'][5],table_4.loc['J'][5],
           table_5.loc['J'][5],table_6.loc['J'][5],table_7.loc['J'][5],table_8.loc['J'][5],
           table_9.loc['J'][5],table_10.loc['J'][5]])

a61=np.nanmean([table_1.loc['K'].accuracy,table_2.loc['K'].accuracy,table_3.loc['K'].accuracy,table_4.loc['K'].accuracy,
           table_5.loc['K'].accuracy,table_6.loc['K'].accuracy,table_7.loc['K'].accuracy,table_8.loc['K'].accuracy,
           table_9.loc['K'].accuracy,table_10.loc['K'].accuracy])


a62=np.nanmean([table_1.loc['K'].f1_score,table_2.loc['K'].f1_score,table_3.loc['K'].f1_score,table_4.loc['K'].f1_score,
           table_5.loc['K'].f1_score,table_6.loc['K'].f1_score,table_7.loc['K'].f1_score,table_8.loc['K'].f1_score,
           table_9.loc['K'].f1_score,table_10.loc['K'].f1_score])


a63=np.nanmean([table_1.loc['K'][2],table_2.loc['K'][2],table_3.loc['K'][2],table_4.loc['K'][2],
           table_5.loc['K'][2],table_6.loc['K'][2],table_7.loc['K'][2],table_8.loc['K'][2],
           table_9.loc['K'][2],table_10.loc['K'][2]])


a64=np.nanmean([table_1.loc['K'][3],table_2.loc['K'][3],table_3.loc['K'][3],table_4.loc['K'][3],
           table_5.loc['K'][3],table_6.loc['K'][3],table_7.loc['K'][3],table_8.loc['K'][3],
           table_9.loc['K'][3],table_10.loc['K'][3]])

a65=np.nanmean([table_1.loc['K'][4],table_2.loc['K'][4],table_3.loc['K'][4],table_4.loc['K'][4],
           table_5.loc['K'][4],table_6.loc['K'][4],table_7.loc['K'][4],table_8.loc['K'][4],
           table_9.loc['K'][4],table_10.loc['K'][4]])

a66=np.nanmean([table_1.loc['K'][5],table_2.loc['K'][5],table_3.loc['K'][5],table_4.loc['K'][5],
           table_5.loc['K'][5],table_6.loc['K'][5],table_7.loc['K'][5],table_8.loc['K'][5],
           table_9.loc['K'][5],table_10.loc['K'][5]])

a67=np.nanmean([table_1.loc['L'].accuracy,table_2.loc['L'].accuracy,table_3.loc['L'].accuracy,table_4.loc['L'].accuracy,
           table_5.loc['L'].accuracy,table_6.loc['L'].accuracy,table_7.loc['L'].accuracy,table_8.loc['L'].accuracy,
           table_9.loc['L'].accuracy,table_10.loc['L'].accuracy])


a68=np.nanmean([table_1.loc['L'].f1_score,table_2.loc['L'].f1_score,table_3.loc['L'].f1_score,table_4.loc['L'].f1_score,
           table_5.loc['L'].f1_score,table_6.loc['L'].f1_score,table_7.loc['L'].f1_score,table_8.loc['L'].f1_score,
           table_9.loc['L'].f1_score,table_10.loc['L'].f1_score])


a69=np.nanmean([table_1.loc['L'][2],table_2.loc['L'][2],table_3.loc['L'][2],table_4.loc['L'][2],
           table_5.loc['L'][2],table_6.loc['L'][2],table_7.loc['L'][2],table_8.loc['L'][2],
           table_9.loc['L'][2],table_10.loc['L'][2]])


a70=np.nanmean([table_1.loc['L'][3],table_2.loc['L'][3],table_3.loc['L'][3],table_4.loc['L'][3],
           table_5.loc['L'][3],table_6.loc['L'][3],table_7.loc['L'][3],table_8.loc['L'][3],
           table_9.loc['L'][3],table_10.loc['L'][3]])

a71=np.nanmean([table_1.loc['L'][4],table_2.loc['L'][4],table_3.loc['L'][4],table_4.loc['L'][4],
           table_5.loc['L'][4],table_6.loc['L'][4],table_7.loc['L'][4],table_8.loc['L'][4],
           table_9.loc['L'][4],table_10.loc['L'][4]])

a72=np.nanmean([table_1.loc['L'][5],table_2.loc['L'][5],table_3.loc['L'][5],table_4.loc['L'][5],
           table_5.loc['L'][5],table_6.loc['L'][5],table_7.loc['L'][5],table_8.loc['L'][5],
           table_9.loc['L'][5],table_10.loc['L'][5]])


a73=np.nanmean([table_1.loc['M'].accuracy,table_2.loc['M'].accuracy,table_3.loc['M'].accuracy,table_4.loc['M'].accuracy,
           table_5.loc['M'].accuracy,table_6.loc['M'].accuracy,table_7.loc['M'].accuracy,table_8.loc['M'].accuracy,
           table_9.loc['M'].accuracy,table_10.loc['M'].accuracy])


a74=np.nanmean([table_1.loc['M'].f1_score,table_2.loc['M'].f1_score,table_3.loc['M'].f1_score,table_4.loc['M'].f1_score,
           table_5.loc['M'].f1_score,table_6.loc['M'].f1_score,table_7.loc['M'].f1_score,table_8.loc['M'].f1_score,
           table_9.loc['M'].f1_score,table_10.loc['M'].f1_score])


a75=np.nanmean([table_1.loc['M'][2],table_2.loc['M'][2],table_3.loc['M'][2],table_4.loc['M'][2],
           table_5.loc['M'][2],table_6.loc['M'][2],table_7.loc['M'][2],table_8.loc['M'][2],
           table_9.loc['M'][2],table_10.loc['M'][2]])


a76=np.nanmean([table_1.loc['M'][3],table_2.loc['M'][3],table_3.loc['M'][3],table_4.loc['M'][3],
           table_5.loc['M'][3],table_6.loc['M'][3],table_7.loc['M'][3],table_8.loc['M'][3],
           table_9.loc['M'][3],table_10.loc['M'][3]])

a77=np.nanmean([table_1.loc['M'][4],table_2.loc['M'][4],table_3.loc['M'][4],table_4.loc['M'][4],
           table_5.loc['M'][4],table_6.loc['M'][4],table_7.loc['M'][4],table_8.loc['M'][4],
           table_9.loc['M'][4],table_10.loc['M'][4]])

a78=np.nanmean([table_1.loc['M'][5],table_2.loc['M'][5],table_3.loc['M'][5],table_4.loc['M'][5],
           table_5.loc['M'][5],table_6.loc['M'][5],table_7.loc['M'][5],table_8.loc['M'][5],
           table_9.loc['M'][5],table_10.loc['M'][5]])


a79=np.nanmean([table_1.loc['.'].accuracy,table_2.loc['.'].accuracy,table_3.loc['.'].accuracy,table_4.loc['.'].accuracy,
           table_5.loc['.'].accuracy,table_6.loc['.'].accuracy,table_7.loc['.'].accuracy,table_8.loc['.'].accuracy,
           table_9.loc['.'].accuracy,table_10.loc['.'].accuracy])


a80=np.nanmean([table_1.loc['.'].f1_score,table_2.loc['.'].f1_score,table_3.loc['.'].f1_score,table_4.loc['.'].f1_score,
           table_5.loc['.'].f1_score,table_6.loc['.'].f1_score,table_7.loc['.'].f1_score,table_8.loc['.'].f1_score,
           table_9.loc['.'].f1_score,table_10.loc['.'].f1_score])


a81=np.nanmean([table_1.loc['.'][2],table_2.loc['.'][2],table_3.loc['.'][2],table_4.loc['.'][2],
           table_5.loc['.'][2],table_6.loc['.'][2],table_7.loc['.'][2],table_8.loc['.'][2],
           table_9.loc['.'][2],table_10.loc['.'][2]])


a82=np.nanmean([table_1.loc['.'][3],table_2.loc['.'][3],table_3.loc['.'][3],table_4.loc['.'][3],
           table_5.loc['.'][3],table_6.loc['.'][3],table_7.loc['.'][3],table_8.loc['.'][3],
           table_9.loc['.'][3],table_10.loc['.'][3]])

a83=np.nanmean([table_1.loc['.'][4],table_2.loc['.'][4],table_3.loc['.'][4],table_4.loc['.'][4],
           table_5.loc['.'][4],table_6.loc['.'][4],table_7.loc['.'][4],table_8.loc['.'][4],
           table_9.loc['.'][4],table_10.loc['.'][4]])

a84=np.nanmean([table_1.loc['.'][5],table_2.loc['.'][5],table_3.loc['.'][5],table_4.loc['.'][5],
           table_5.loc['.'][5],table_6.loc['.'][5],table_7.loc['.'][5],table_8.loc['.'][5],
           table_9.loc['.'][5],table_10.loc['.'][5]])


A=[[a1,a2,a3,round(a4),a5,round(a6)],[a7,a8,a9,round(a10),a11,round(a12)],[a13,a14,a15,round(a16),a17,round(a18)],
    [a19,a20,a21,round(a22),a23,round(a24)]
,[a25,a26,a27,round(a28),a29,round(a30)],[a31,a32,a33,round(a34),a35,round(a36)],
[a37,a38,a39,round(a40),a41,round(a42)],[a43,a44,a45,round(a46),a47,round(a48)],
[a49,a50,a51,round(a52),a53,round(a54)],[a55,a56,a57,round(a58),a59,round(a60)],
[a61,a62,a63,round(a64),a65,round(a66)],[a67,a68,a69,round(a70),a71,round(a72)],
[a73,a74,a75,round(a76),a77,round(a78)],[a79,a80,a81,round(a82),a83,round(a84)]]

vv1=np.nanmean([v1[0],v2[0],v3[0],v4[0],v5[0],v6[0],v7[0],v8[0],v9[0],v10[0]])
vv2=np.nanmean([v1[1],v2[1],v3[1],v4[1],v5[1],v6[1],v7[1],v8[1],v9[1],v10[1]])
vv3=np.nanmean([v1[2],v2[2],v3[2],v4[2],v5[2],v6[2],v7[2],v8[2],v9[2],v10[2]])
vv4=np.nanmean([v1[3],v2[3],v3[3],v4[3],v5[3],v6[3],v7[3],v8[3],v9[3],v10[3]])
vv5=np.nanmean([v1[4],v2[4],v3[4],v4[4],v5[4],v6[4],v7[4],v8[4],v9[4],v10[4]])
vv6=np.nanmean([v1[5],v2[5],v3[5],v4[5],v5[5],v6[5],v7[5],v8[5],v9[5],v10[5]])
table_11= pd.DataFrame(A,columns=['accuracy', 'f1_score', 'accuracy for unknown words',
'number of unknown words','accuracy for known words','number of known words']
,index=['A','B','C','D','E','F','G','H','I','J','K','L','M','.'])

#table_10= pd.DataFrame(A,
#columns=['accuracy', 'f1_score', 'accuracy for unknown words',
#         'number of unknown words','accuracy for known words','number of known words']
#,index=[list(tag2idx.keys())[0], list(tag2idx.keys())[1], list(tag2idx.keys())[2] , list(tag2idx.keys())[3] 
#, list(tag2idx.keys())[4] , list(tag2idx.keys())[5],list(tag2idx.keys())[6],list(tag2idx.keys())[7]
#,list(tag2idx.keys())[8],list(tag2idx.keys())[9],list(tag2idx.keys())[10],list(tag2idx.keys())[11],
#list(tag2idx.keys())[12],list(tag2idx.keys())[13]])

str_pythontex=[float("{0:.2f}".format(list(table_11.loc["A"])[0]*100)),float("{0:.2f}".format(list(table_11.loc["A"])[1]*100)),
float("{0:.2f}".format(list(table_11.loc["A"])[2]*100)),float("{0:.2f}".format(list(table_11.loc["A"])[4]*100)),
round(list(table_11.loc["A"])[3]),round(list(table_11.loc["A"])[5]),
float("{0:.2f}".format(list(table_11.loc["B"])[0]*100)),float("{0:.2f}".format(list(table_11.loc["B"])[1]*100)),
float("{0:.2f}".format(list(table_11.loc["B"])[2]*100)),float("{0:.2f}".format(list(table_11.loc["B"])[4]*100)),
round(list(table_11.loc["B"])[3]),round(list(table_11.loc["B"])[5]),
float("{0:.2f}".format(list(table_11.loc["C"])[0]*100)),float("{0:.2f}".format(list(table_11.loc["C"])[1]*100)),
float("{0:.2f}".format(list(table_11.loc["C"])[2]*100)),float("{0:.2f}".format(list(table_11.loc["C"])[4]*100)),
round(list(table_11.loc["C"])[3]),round(list(table_11.loc["C"])[5]),
float("{0:.2f}".format(list(table_11.loc["D"])[0]*100)),float("{0:.2f}".format(list(table_11.loc["D"])[1]*100)),
float("{0:.2f}".format(list(table_11.loc["D"])[2]*100)),float("{0:.2f}".format(list(table_11.loc["D"])[4]*100)),
round(list(table_11.loc["D"])[3]),round(list(table_11.loc["D"])[5]),
float("{0:.2f}".format(list(table_11.loc["E"])[0]*100)),float("{0:.2f}".format(list(table_11.loc["E"])[1]*100)),
float("{0:.2f}".format(list(table_11.loc["E"])[2]*100)),float("{0:.2f}".format(list(table_11.loc["E"])[4]*100)),
round(list(table_11.loc["E"])[3]),round(list(table_11.loc["E"])[5]),
float("{0:.2f}".format(list(table_11.loc["F"])[0]*100)),float("{0:.2f}".format(list(table_11.loc["F"])[1]*100)),
float("{0:.2f}".format(list(table_11.loc["F"])[2]*100)),float("{0:.2f}".format(list(table_11.loc["F"])[4]*100)),
round(list(table_11.loc["F"])[3]),round(list(table_11.loc["F"])[5]),
float("{0:.2f}".format(list(table_11.loc["G"])[0]*100)),float("{0:.2f}".format(list(table_11.loc["G"])[1]*100)),
float("{0:.2f}".format(list(table_11.loc["G"])[2]*100)),float("{0:.2f}".format(list(table_11.loc["G"])[4]*100)),
round(list(table_11.loc["G"])[3]),round(list(table_11.loc["G"])[5]),
float("{0:.2f}".format(list(table_11.loc["H"])[0]*100)),float("{0:.2f}".format(list(table_11.loc["H"])[1]*100)),
float("{0:.2f}".format(list(table_11.loc["H"])[2]*100)),float("{0:.2f}".format(list(table_11.loc["H"])[4]*100)),
round(list(table_11.loc["H"])[3]),round(list(table_11.loc["H"])[5]),
float("{0:.2f}".format(list(table_11.loc["I"])[0]*100)),float("{0:.2f}".format(list(table_11.loc["I"])[1]*100)),
float("{0:.2f}".format(list(table_11.loc["I"])[2]*100)),float("{0:.2f}".format(list(table_11.loc["I"])[4]*100)),
round(list(table_11.loc["I"])[3]),round(list(table_11.loc["I"])[5]),
float("{0:.2f}".format(list(table_11.loc["J"])[0]*100)),float("{0:.2f}".format(list(table_11.loc["J"])[1]*100)),
float("{0:.2f}".format(list(table_11.loc["J"])[2]*100)),float("{0:.2f}".format(list(table_11.loc["J"])[4]*100)),
round(list(table_11.loc["J"])[3]),round(list(table_11.loc["J"])[5]),
float("{0:.2f}".format(list(table_11.loc["K"])[0]*100)),float("{0:.2f}".format(list(table_11.loc["K"])[1]*100)),
float("{0:.2f}".format(list(table_11.loc["K"])[2]*100)),float("{0:.2f}".format(list(table_11.loc["K"])[4]*100)),
round(list(table_11.loc["K"])[3]),round(list(table_11.loc["K"])[5]),
float("{0:.2f}".format(list(table_11.loc["L"])[0]*100)),float("{0:.2f}".format(list(table_11.loc["L"])[1]*100)),
float("{0:.2f}".format(list(table_11.loc["L"])[2]*100)),float("{0:.2f}".format(list(table_11.loc["L"])[4]*100)),
round(list(table_11.loc["L"])[3]),round(list(table_11.loc["L"])[5]),
float("{0:.2f}".format(list(table_11.loc["M"])[0]*100)),float("{0:.2f}".format(list(table_11.loc["M"])[1]*100)),
float("{0:.2f}".format(list(table_11.loc["M"])[2]*100)),float("{0:.2f}".format(list(table_11.loc["M"])[4]*100)),
round(list(table_11.loc["M"])[3]),round(list(table_11.loc["M"])[5]),
float("{0:.2f}".format(list(table_11.loc["."])[0]*100)),float("{0:.2f}".format(list(table_11.loc["."])[1]*100)),
float("{0:.2f}".format(list(table_11.loc["."])[2]*100)),float("{0:.2f}".format(list(table_11.loc["."])[4]*100)),
round(list(table_11.loc["."])[3]),round(list(table_11.loc["."])[5]),float("{0:.2f}".format(vv1))
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

