# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 09:49:40 2020

@author: Mehdi
"""

import numpy as np

a1=np.nanmean([table_11.loc['A'].accuracy,table_12.loc['A'].accuracy,table_13.loc['A'].accuracy,table_14.loc['A'].accuracy,
           table_15.loc['A'].accuracy,table_16.loc['A'].accuracy,table_17.loc['A'].accuracy,table_18.loc['A'].accuracy,
           table_19.loc['A'].accuracy,table_110.loc['A'].accuracy])


a2=np.nanmean([table_11.loc['A'].f1_score,table_12.loc['A'].f1_score,table_13.loc['A'].f1_score,table_14.loc['A'].f1_score,
           table_15.loc['A'].f1_score,table_16.loc['A'].f1_score,table_17.loc['A'].f1_score,table_18.loc['A'].f1_score,
           table_19.loc['A'].f1_score,table_110.loc['A'].f1_score])


a3=np.nanmean([table_11.loc['A'][2],table_12.loc['A'][2],table_13.loc['A'][2],table_14.loc['A'][2],
           table_15.loc['A'][2],table_16.loc['A'][2],table_17.loc['A'][2],table_18.loc['A'][2],
           table_19.loc['A'][2],table_110.loc['A'][2]])


a4=np.nanmean([table_11.loc['A'][3],table_12.loc['A'][3],table_13.loc['A'][3],table_14.loc['A'][3],
           table_15.loc['A'][3],table_16.loc['A'][3],table_17.loc['A'][3],table_18.loc['A'][3],
           table_19.loc['A'][3],table_110.loc['A'][3]])

a5=np.nanmean([table_11.loc['A'][4],table_12.loc['A'][4],table_13.loc['A'][4],table_14.loc['A'][4],
           table_15.loc['A'][4],table_16.loc['A'][4],table_17.loc['A'][4],table_18.loc['A'][4],
           table_19.loc['A'][4],table_110.loc['A'][4]])

a6=np.nanmean([table_11.loc['A'][5],table_12.loc['A'][5],table_13.loc['A'][5],table_14.loc['A'][5],
           table_15.loc['A'][5],table_16.loc['A'][5],table_17.loc['A'][5],table_18.loc['A'][5],
           table_19.loc['A'][5],table_110.loc['A'][5]])

a7=np.nanmean([table_11.loc['B'].accuracy,table_12.loc['B'].accuracy,table_13.loc['B'].accuracy,table_14.loc['B'].accuracy,
           table_15.loc['B'].accuracy,table_16.loc['B'].accuracy,table_17.loc['B'].accuracy,table_18.loc['B'].accuracy,
           table_19.loc['B'].accuracy,table_110.loc['B'].accuracy])


a8=np.nanmean([table_11.loc['B'].f1_score,table_12.loc['B'].f1_score,table_13.loc['B'].f1_score,table_14.loc['B'].f1_score,
           table_15.loc['B'].f1_score,table_16.loc['B'].f1_score,table_17.loc['B'].f1_score,table_18.loc['B'].f1_score,
           table_19.loc['B'].f1_score,table_110.loc['B'].f1_score])


a9=np.nanmean([table_11.loc['B'][2],table_12.loc['B'][2],table_13.loc['B'][2],table_14.loc['B'][2],
           table_15.loc['B'][2],table_16.loc['B'][2],table_17.loc['B'][2],table_18.loc['B'][2],
           table_19.loc['B'][2],table_110.loc['B'][2]])


a10=np.nanmean([table_11.loc['B'][3],table_12.loc['B'][3],table_13.loc['B'][3],table_14.loc['B'][3],
           table_15.loc['B'][3],table_16.loc['B'][3],table_17.loc['B'][3],table_18.loc['B'][3],
           table_19.loc['B'][3],table_110.loc['B'][3]])

a11=np.nanmean([table_11.loc['B'][4],table_12.loc['B'][4],table_13.loc['B'][4],table_14.loc['B'][4],
           table_15.loc['B'][4],table_16.loc['B'][4],table_17.loc['B'][4],table_18.loc['B'][4],
           table_19.loc['B'][4],table_110.loc['B'][4]])

a12=np.nanmean([table_11.loc['B'][5],table_12.loc['B'][5],table_13.loc['B'][5],table_14.loc['B'][5],
           table_15.loc['B'][5],table_16.loc['B'][5],table_17.loc['B'][5],table_18.loc['B'][5],
           table_19.loc['B'][5],table_110.loc['B'][5]])


a13=np.nanmean([table_11.loc['C'].accuracy,table_12.loc['C'].accuracy,table_13.loc['C'].accuracy,table_14.loc['C'].accuracy,
           table_15.loc['C'].accuracy,table_16.loc['C'].accuracy,table_17.loc['C'].accuracy,table_18.loc['C'].accuracy,
           table_19.loc['C'].accuracy,table_110.loc['C'].accuracy])


a14=np.nanmean([table_11.loc['C'].f1_score,table_12.loc['C'].f1_score,table_13.loc['C'].f1_score,table_14.loc['C'].f1_score,
           table_15.loc['C'].f1_score,table_16.loc['C'].f1_score,table_17.loc['C'].f1_score,table_18.loc['C'].f1_score,
           table_19.loc['C'].f1_score,table_110.loc['C'].f1_score])


a15=np.nanmean([table_11.loc['C'][2],table_12.loc['C'][2],table_13.loc['C'][2],table_14.loc['C'][2],
           table_15.loc['C'][2],table_16.loc['C'][2],table_17.loc['C'][2],table_18.loc['C'][2],
           table_19.loc['C'][2],table_110.loc['C'][2]])


a16=np.nanmean([table_11.loc['C'][3],table_12.loc['C'][3],table_13.loc['C'][3],table_14.loc['C'][3],
           table_15.loc['C'][3],table_16.loc['C'][3],table_17.loc['C'][3],table_18.loc['C'][3],
           table_19.loc['C'][3],table_110.loc['C'][3]])

a17=np.nanmean([table_11.loc['C'][4],table_12.loc['C'][4],table_13.loc['C'][4],table_14.loc['C'][4],
           table_15.loc['C'][4],table_16.loc['C'][4],table_17.loc['C'][4],table_18.loc['C'][4],
           table_19.loc['C'][4],table_110.loc['C'][4]])

a18=np.nanmean([table_11.loc['C'][5],table_12.loc['C'][5],table_13.loc['C'][5],table_14.loc['C'][5],
           table_15.loc['C'][5],table_16.loc['C'][5],table_17.loc['C'][5],table_18.loc['C'][5],
           table_19.loc['C'][5],table_110.loc['C'][5]])

a19=np.nanmean([table_11.loc['D'].accuracy,table_12.loc['D'].accuracy,table_13.loc['D'].accuracy,table_14.loc['D'].accuracy,
           table_15.loc['D'].accuracy,table_16.loc['D'].accuracy,table_17.loc['D'].accuracy,table_18.loc['D'].accuracy,
           table_19.loc['D'].accuracy,table_110.loc['D'].accuracy])


a20=np.nanmean([table_11.loc['D'].f1_score,table_12.loc['D'].f1_score,table_13.loc['D'].f1_score,table_14.loc['D'].f1_score,
           table_15.loc['D'].f1_score,table_16.loc['D'].f1_score,table_17.loc['D'].f1_score,table_18.loc['D'].f1_score,
           table_19.loc['D'].f1_score,table_110.loc['D'].f1_score])


a21=np.nanmean([table_11.loc['D'][2],table_12.loc['D'][2],table_13.loc['D'][2],table_14.loc['D'][2],
           table_15.loc['D'][2],table_16.loc['D'][2],table_17.loc['D'][2],table_18.loc['D'][2],
           table_19.loc['D'][2],table_110.loc['D'][2]])


a22=np.nanmean([table_11.loc['D'][3],table_12.loc['D'][3],table_13.loc['D'][3],table_14.loc['D'][3],
           table_15.loc['D'][3],table_16.loc['D'][3],table_17.loc['D'][3],table_18.loc['D'][3],
           table_19.loc['D'][3],table_110.loc['D'][3]])

a23=np.nanmean([table_11.loc['D'][4],table_12.loc['D'][4],table_13.loc['D'][4],table_14.loc['D'][4],
           table_15.loc['D'][4],table_16.loc['D'][4],table_17.loc['D'][4],table_18.loc['D'][4],
           table_19.loc['D'][4],table_110.loc['D'][4]])

a24=np.nanmean([table_11.loc['D'][5],table_12.loc['D'][5],table_13.loc['D'][5],table_14.loc['D'][5],
           table_15.loc['D'][5],table_16.loc['D'][5],table_17.loc['D'][5],table_18.loc['D'][5],
           table_19.loc['D'][5],table_110.loc['D'][5]])


a25=np.nanmean([table_11.loc['E'].accuracy,table_12.loc['E'].accuracy,table_13.loc['E'].accuracy,table_14.loc['E'].accuracy,
           table_15.loc['E'].accuracy,table_16.loc['E'].accuracy,table_17.loc['E'].accuracy,table_18.loc['E'].accuracy,
           table_19.loc['E'].accuracy,table_110.loc['E'].accuracy])


a26=np.nanmean([table_11.loc['E'].f1_score,table_12.loc['E'].f1_score,table_13.loc['E'].f1_score,table_14.loc['E'].f1_score,
           table_15.loc['E'].f1_score,table_16.loc['E'].f1_score,table_17.loc['E'].f1_score,table_18.loc['E'].f1_score,
           table_19.loc['E'].f1_score,table_110.loc['E'].f1_score])


a27=np.nanmean([table_11.loc['E'][2],table_12.loc['E'][2],table_13.loc['E'][2],table_14.loc['E'][2],
           table_15.loc['E'][2],table_16.loc['E'][2],table_17.loc['E'][2],table_18.loc['E'][2],
           table_19.loc['E'][2],table_110.loc['E'][2]])


a28=np.nanmean([table_11.loc['E'][3],table_12.loc['E'][3],table_13.loc['E'][3],table_14.loc['E'][3],
           table_15.loc['E'][3],table_16.loc['E'][3],table_17.loc['E'][3],table_18.loc['E'][3],
           table_19.loc['E'][3],table_110.loc['E'][3]])

a29=np.nanmean([table_11.loc['E'][4],table_12.loc['E'][4],table_13.loc['E'][4],table_14.loc['E'][4],
           table_15.loc['E'][4],table_16.loc['E'][4],table_17.loc['E'][4],table_18.loc['E'][4],
           table_19.loc['E'][4],table_110.loc['E'][4]])

a30=np.nanmean([table_11.loc['E'][5],table_12.loc['E'][5],table_13.loc['E'][5],table_14.loc['E'][5],
           table_15.loc['E'][5],table_16.loc['E'][5],table_17.loc['E'][5],table_18.loc['E'][5],
           table_19.loc['E'][5],table_110.loc['E'][5]])


a31=np.nanmean([table_11.loc['F'].accuracy,table_12.loc['F'].accuracy,table_13.loc['F'].accuracy,table_14.loc['F'].accuracy,
           table_15.loc['F'].accuracy,table_16.loc['F'].accuracy,table_17.loc['F'].accuracy,table_18.loc['F'].accuracy,
           table_19.loc['F'].accuracy,table_110.loc['F'].accuracy])


a32=np.nanmean([table_11.loc['F'].f1_score,table_12.loc['F'].f1_score,table_13.loc['F'].f1_score,table_14.loc['F'].f1_score,
           table_15.loc['F'].f1_score,table_16.loc['F'].f1_score,table_17.loc['F'].f1_score,table_18.loc['F'].f1_score,
           table_19.loc['F'].f1_score,table_110.loc['F'].f1_score])


a33=np.nanmean([table_11.loc['F'][2],table_12.loc['F'][2],table_13.loc['F'][2],table_14.loc['F'][2],
           table_15.loc['F'][2],table_16.loc['F'][2],table_17.loc['F'][2],table_18.loc['F'][2],
           table_19.loc['F'][2],table_110.loc['F'][2]])


a34=np.nanmean([table_11.loc['F'][3],table_12.loc['F'][3],table_13.loc['F'][3],table_14.loc['F'][3],
           table_15.loc['F'][3],table_16.loc['F'][3],table_17.loc['F'][3],table_18.loc['F'][3],
           table_19.loc['F'][3],table_110.loc['F'][3]])

a35=np.nanmean([table_11.loc['F'][4],table_12.loc['F'][4],table_13.loc['F'][4],table_14.loc['F'][4],
           table_15.loc['F'][4],table_16.loc['F'][4],table_17.loc['F'][4],table_18.loc['F'][4],
           table_19.loc['F'][4],table_110.loc['F'][4]])

a36=np.nanmean([table_11.loc['F'][5],table_12.loc['F'][5],table_13.loc['F'][5],table_14.loc['F'][5],
           table_15.loc['F'][5],table_16.loc['F'][5],table_17.loc['F'][5],table_18.loc['F'][5],
           table_19.loc['F'][5],table_110.loc['F'][5]])

a37=np.nanmean([table_11.loc['G'].accuracy,table_12.loc['G'].accuracy,table_13.loc['G'].accuracy,table_14.loc['G'].accuracy,
           table_15.loc['G'].accuracy,table_16.loc['G'].accuracy,table_17.loc['G'].accuracy,table_18.loc['G'].accuracy,
           table_19.loc['G'].accuracy,table_110.loc['G'].accuracy])


a38=np.nanmean([table_11.loc['G'].f1_score,table_12.loc['G'].f1_score,table_13.loc['G'].f1_score,table_14.loc['G'].f1_score,
           table_15.loc['G'].f1_score,table_16.loc['G'].f1_score,table_17.loc['G'].f1_score,table_18.loc['G'].f1_score,
           table_19.loc['G'].f1_score,table_110.loc['G'].f1_score])


a39=np.nanmean([table_11.loc['G'][2],table_12.loc['G'][2],table_13.loc['G'][2],table_14.loc['G'][2],
           table_15.loc['G'][2],table_16.loc['G'][2],table_17.loc['G'][2],table_18.loc['G'][2],
           table_19.loc['G'][2],table_110.loc['G'][2]])


a40=np.nanmean([table_11.loc['G'][3],table_12.loc['G'][3],table_13.loc['G'][3],table_14.loc['G'][3],
           table_15.loc['G'][3],table_16.loc['G'][3],table_17.loc['G'][3],table_18.loc['G'][3],
           table_19.loc['G'][3],table_110.loc['G'][3]])

a41=np.nanmean([table_11.loc['G'][4],table_12.loc['G'][4],table_13.loc['G'][4],table_14.loc['G'][4],
           table_15.loc['G'][4],table_16.loc['G'][4],table_17.loc['G'][4],table_18.loc['G'][4],
           table_19.loc['G'][4],table_110.loc['G'][4]])

a42=np.nanmean([table_11.loc['G'][5],table_12.loc['G'][5],table_13.loc['G'][5],table_14.loc['G'][5],
           table_15.loc['G'][5],table_16.loc['G'][5],table_17.loc['G'][5],table_18.loc['G'][5],
           table_19.loc['G'][5],table_110.loc['G'][5]])


a43=np.nanmean([table_11.loc['H'].accuracy,table_12.loc['H'].accuracy,table_13.loc['H'].accuracy,table_14.loc['H'].accuracy,
           table_15.loc['H'].accuracy,table_16.loc['H'].accuracy,table_17.loc['H'].accuracy,table_18.loc['H'].accuracy,
           table_19.loc['H'].accuracy,table_110.loc['H'].accuracy])


a44=np.nanmean([table_11.loc['H'].f1_score,table_12.loc['H'].f1_score,table_13.loc['H'].f1_score,table_14.loc['H'].f1_score,
           table_15.loc['H'].f1_score,table_16.loc['H'].f1_score,table_17.loc['H'].f1_score,table_18.loc['H'].f1_score,
           table_19.loc['H'].f1_score,table_110.loc['H'].f1_score])


a45=np.nanmean([table_11.loc['H'][2],table_12.loc['H'][2],table_13.loc['H'][2],table_14.loc['H'][2],
           table_15.loc['H'][2],table_16.loc['H'][2],table_17.loc['H'][2],table_18.loc['H'][2],
           table_19.loc['H'][2],table_110.loc['H'][2]])


a46=np.nanmean([table_11.loc['H'][3],table_12.loc['H'][3],table_13.loc['H'][3],table_14.loc['H'][3],
           table_15.loc['H'][3],table_16.loc['H'][3],table_17.loc['H'][3],table_18.loc['H'][3],
           table_19.loc['H'][3],table_110.loc['H'][3]])

a47=np.nanmean([table_11.loc['H'][4],table_12.loc['H'][4],table_13.loc['H'][4],table_14.loc['H'][4],
           table_15.loc['H'][4],table_16.loc['H'][4],table_17.loc['H'][4],table_18.loc['H'][4],
           table_19.loc['H'][4],table_110.loc['H'][4]])

a48=np.nanmean([table_11.loc['H'][5],table_12.loc['H'][5],table_13.loc['H'][5],table_14.loc['H'][5],
           table_15.loc['H'][5],table_16.loc['H'][5],table_17.loc['H'][5],table_18.loc['H'][5],
           table_19.loc['H'][5],table_110.loc['H'][5]])


a49=np.nanmean([table_11.loc['I'].accuracy,table_12.loc['I'].accuracy,table_13.loc['I'].accuracy,table_14.loc['I'].accuracy,
           table_15.loc['I'].accuracy,table_16.loc['I'].accuracy,table_17.loc['I'].accuracy,table_18.loc['I'].accuracy,
           table_19.loc['I'].accuracy,table_110.loc['I'].accuracy])


a50=np.nanmean([table_11.loc['I'].f1_score,table_12.loc['I'].f1_score,table_13.loc['I'].f1_score,table_14.loc['I'].f1_score,
           table_15.loc['I'].f1_score,table_16.loc['I'].f1_score,table_17.loc['I'].f1_score,table_18.loc['I'].f1_score,
           table_19.loc['I'].f1_score,table_110.loc['I'].f1_score])


a51=np.nanmean([table_11.loc['I'][2],table_12.loc['I'][2],table_13.loc['I'][2],table_14.loc['I'][2],
           table_15.loc['I'][2],table_16.loc['I'][2],table_17.loc['I'][2],table_18.loc['I'][2],
           table_19.loc['I'][2],table_110.loc['I'][2]])


a52=np.nanmean([table_11.loc['I'][3],table_12.loc['I'][3],table_13.loc['I'][3],table_14.loc['I'][3],
           table_15.loc['I'][3],table_16.loc['I'][3],table_17.loc['I'][3],table_18.loc['I'][3],
           table_19.loc['I'][3],table_110.loc['I'][3]])

a53=np.nanmean([table_11.loc['I'][4],table_12.loc['I'][4],table_13.loc['I'][4],table_14.loc['I'][4],
           table_15.loc['I'][4],table_16.loc['I'][4],table_17.loc['I'][4],table_18.loc['I'][4],
           table_19.loc['I'][4],table_110.loc['I'][4]])

a54=np.nanmean([table_11.loc['I'][5],table_12.loc['I'][5],table_13.loc['I'][5],table_14.loc['I'][5],
           table_15.loc['I'][5],table_16.loc['I'][5],table_17.loc['I'][5],table_18.loc['I'][5],
           table_19.loc['I'][5],table_110.loc['I'][5]])


a55=np.nanmean([table_11.loc['J'].accuracy,table_12.loc['J'].accuracy,table_13.loc['J'].accuracy,table_14.loc['J'].accuracy,
           table_15.loc['J'].accuracy,table_16.loc['J'].accuracy,table_17.loc['J'].accuracy,table_18.loc['J'].accuracy,
           table_19.loc['J'].accuracy,table_110.loc['J'].accuracy])


a56=np.nanmean([table_11.loc['J'].f1_score,table_12.loc['J'].f1_score,table_13.loc['J'].f1_score,table_14.loc['J'].f1_score,
           table_15.loc['J'].f1_score,table_16.loc['J'].f1_score,table_17.loc['J'].f1_score,table_18.loc['J'].f1_score,
           table_19.loc['J'].f1_score,table_110.loc['J'].f1_score])


a57=np.nanmean([table_11.loc['J'][2],table_12.loc['J'][2],table_13.loc['J'][2],table_14.loc['J'][2],
           table_15.loc['J'][2],table_16.loc['J'][2],table_17.loc['J'][2],table_18.loc['J'][2],
           table_19.loc['J'][2],table_110.loc['J'][2]])


a58=np.nanmean([table_11.loc['J'][3],table_12.loc['J'][3],table_13.loc['J'][3],table_14.loc['J'][3],
           table_15.loc['J'][3],table_16.loc['J'][3],table_17.loc['J'][3],table_18.loc['J'][3],
           table_19.loc['J'][3],table_110.loc['J'][3]])

a59=np.nanmean([table_11.loc['J'][4],table_12.loc['J'][4],table_13.loc['J'][4],table_14.loc['J'][4],
           table_15.loc['J'][4],table_16.loc['J'][4],table_17.loc['J'][4],table_18.loc['J'][4],
           table_19.loc['J'][4],table_110.loc['J'][4]])

a60=np.nanmean([table_11.loc['J'][5],table_12.loc['J'][5],table_13.loc['J'][5],table_14.loc['J'][5],
           table_15.loc['J'][5],table_16.loc['J'][5],table_17.loc['J'][5],table_18.loc['J'][5],
           table_19.loc['J'][5],table_110.loc['J'][5]])

a61=np.nanmean([table_11.loc['K'].accuracy,table_12.loc['K'].accuracy,table_13.loc['K'].accuracy,table_14.loc['K'].accuracy,
           table_15.loc['K'].accuracy,table_16.loc['K'].accuracy,table_17.loc['K'].accuracy,table_18.loc['K'].accuracy,
           table_19.loc['K'].accuracy,table_110.loc['K'].accuracy])


a62=np.nanmean([table_11.loc['K'].f1_score,table_12.loc['K'].f1_score,table_13.loc['K'].f1_score,table_14.loc['K'].f1_score,
           table_15.loc['K'].f1_score,table_16.loc['K'].f1_score,table_17.loc['K'].f1_score,table_18.loc['K'].f1_score,
           table_19.loc['K'].f1_score,table_110.loc['K'].f1_score])


a63=np.nanmean([table_11.loc['K'][2],table_12.loc['K'][2],table_13.loc['K'][2],table_14.loc['K'][2],
           table_15.loc['K'][2],table_16.loc['K'][2],table_17.loc['K'][2],table_18.loc['K'][2],
           table_19.loc['K'][2],table_110.loc['K'][2]])


a64=np.nanmean([table_11.loc['K'][3],table_12.loc['K'][3],table_13.loc['K'][3],table_14.loc['K'][3],
           table_15.loc['K'][3],table_16.loc['K'][3],table_17.loc['K'][3],table_18.loc['K'][3],
           table_19.loc['K'][3],table_110.loc['K'][3]])

a65=np.nanmean([table_11.loc['K'][4],table_12.loc['K'][4],table_13.loc['K'][4],table_14.loc['K'][4],
           table_15.loc['K'][4],table_16.loc['K'][4],table_17.loc['K'][4],table_18.loc['K'][4],
           table_19.loc['K'][4],table_110.loc['K'][4]])

a66=np.nanmean([table_11.loc['K'][5],table_12.loc['K'][5],table_13.loc['K'][5],table_14.loc['K'][5],
           table_15.loc['K'][5],table_16.loc['K'][5],table_17.loc['K'][5],table_18.loc['K'][5],
           table_19.loc['K'][5],table_110.loc['K'][5]])

a67=np.nanmean([table_11.loc['L'].accuracy,table_12.loc['L'].accuracy,table_13.loc['L'].accuracy,table_14.loc['L'].accuracy,
           table_15.loc['L'].accuracy,table_16.loc['L'].accuracy,table_17.loc['L'].accuracy,table_18.loc['L'].accuracy,
           table_19.loc['L'].accuracy,table_110.loc['L'].accuracy])


a68=np.nanmean([table_11.loc['L'].f1_score,table_12.loc['L'].f1_score,table_13.loc['L'].f1_score,table_14.loc['L'].f1_score,
           table_15.loc['L'].f1_score,table_16.loc['L'].f1_score,table_17.loc['L'].f1_score,table_18.loc['L'].f1_score,
           table_19.loc['L'].f1_score,table_110.loc['L'].f1_score])


a69=np.nanmean([table_11.loc['L'][2],table_12.loc['L'][2],table_13.loc['L'][2],table_14.loc['L'][2],
           table_15.loc['L'][2],table_16.loc['L'][2],table_17.loc['L'][2],table_18.loc['L'][2],
           table_19.loc['L'][2],table_110.loc['L'][2]])


a70=np.nanmean([table_11.loc['L'][3],table_12.loc['L'][3],table_13.loc['L'][3],table_14.loc['L'][3],
           table_15.loc['L'][3],table_16.loc['L'][3],table_17.loc['L'][3],table_18.loc['L'][3],
           table_19.loc['L'][3],table_110.loc['L'][3]])

a71=np.nanmean([table_11.loc['L'][4],table_12.loc['L'][4],table_13.loc['L'][4],table_14.loc['L'][4],
           table_15.loc['L'][4],table_16.loc['L'][4],table_17.loc['L'][4],table_18.loc['L'][4],
           table_19.loc['L'][4],table_110.loc['L'][4]])

a72=np.nanmean([table_11.loc['L'][5],table_12.loc['L'][5],table_13.loc['L'][5],table_14.loc['L'][5],
           table_15.loc['L'][5],table_16.loc['L'][5],table_17.loc['L'][5],table_18.loc['L'][5],
           table_19.loc['L'][5],table_110.loc['L'][5]])


a73=np.nanmean([table_11.loc['M'].accuracy,table_12.loc['M'].accuracy,table_13.loc['M'].accuracy,table_14.loc['M'].accuracy,
           table_15.loc['M'].accuracy,table_16.loc['M'].accuracy,table_17.loc['M'].accuracy,table_18.loc['M'].accuracy,
           table_19.loc['M'].accuracy,table_110.loc['M'].accuracy])


a74=np.nanmean([table_11.loc['M'].f1_score,table_12.loc['M'].f1_score,table_13.loc['M'].f1_score,table_14.loc['M'].f1_score,
           table_15.loc['M'].f1_score,table_16.loc['M'].f1_score,table_17.loc['M'].f1_score,table_18.loc['M'].f1_score,
           table_19.loc['M'].f1_score,table_110.loc['M'].f1_score])


a75=np.nanmean([table_11.loc['M'][2],table_12.loc['M'][2],table_13.loc['M'][2],table_14.loc['M'][2],
           table_15.loc['M'][2],table_16.loc['M'][2],table_17.loc['M'][2],table_18.loc['M'][2],
           table_19.loc['M'][2],table_110.loc['M'][2]])


a76=np.nanmean([table_11.loc['M'][3],table_12.loc['M'][3],table_13.loc['M'][3],table_14.loc['M'][3],
           table_15.loc['M'][3],table_16.loc['M'][3],table_17.loc['M'][3],table_18.loc['M'][3],
           table_19.loc['M'][3],table_110.loc['M'][3]])

a77=np.nanmean([table_11.loc['M'][4],table_12.loc['M'][4],table_13.loc['M'][4],table_14.loc['M'][4],
           table_15.loc['M'][4],table_16.loc['M'][4],table_17.loc['M'][4],table_18.loc['M'][4],
           table_19.loc['M'][4],table_110.loc['M'][4]])

a78=np.nanmean([table_11.loc['M'][5],table_12.loc['M'][5],table_13.loc['M'][5],table_14.loc['M'][5],
           table_15.loc['M'][5],table_16.loc['M'][5],table_17.loc['M'][5],table_18.loc['M'][5],
           table_19.loc['M'][5],table_110.loc['M'][5]])


a79=np.nanmean([table_11.loc['.'].accuracy,table_12.loc['.'].accuracy,table_13.loc['.'].accuracy,table_14.loc['.'].accuracy,
           table_15.loc['.'].accuracy,table_16.loc['.'].accuracy,table_17.loc['.'].accuracy,table_18.loc['.'].accuracy,
           table_19.loc['.'].accuracy,table_110.loc['.'].accuracy])


a80=np.nanmean([table_11.loc['.'].f1_score,table_12.loc['.'].f1_score,table_13.loc['.'].f1_score,table_14.loc['.'].f1_score,
           table_15.loc['.'].f1_score,table_16.loc['.'].f1_score,table_17.loc['.'].f1_score,table_18.loc['.'].f1_score,
           table_19.loc['.'].f1_score,table_110.loc['.'].f1_score])


a81=np.nanmean([table_11.loc['.'][2],table_12.loc['.'][2],table_13.loc['.'][2],table_14.loc['.'][2],
           table_15.loc['.'][2],table_16.loc['.'][2],table_17.loc['.'][2],table_18.loc['.'][2],
           table_19.loc['.'][2],table_110.loc['.'][2]])


a82=np.nanmean([table_11.loc['.'][3],table_12.loc['.'][3],table_13.loc['.'][3],table_14.loc['.'][3],
           table_15.loc['.'][3],table_16.loc['.'][3],table_17.loc['.'][3],table_18.loc['.'][3],
           table_19.loc['.'][3],table_110.loc['.'][3]])

a83=np.nanmean([table_11.loc['.'][4],table_12.loc['.'][4],table_13.loc['.'][4],table_14.loc['.'][4],
           table_15.loc['.'][4],table_16.loc['.'][4],table_17.loc['.'][4],table_18.loc['.'][4],
           table_19.loc['.'][4],table_110.loc['.'][4]])

a84=np.nanmean([table_11.loc['.'][5],table_12.loc['.'][5],table_13.loc['.'][5],table_14.loc['.'][5],
           table_15.loc['.'][5],table_16.loc['.'][5],table_17.loc['.'][5],table_18.loc['.'][5],
           table_19.loc['.'][5],table_110.loc['.'][5]])


A=[[a1,a2,a3,round(a4),a5,round(a6)],[a7,a8,a9,round(a10),a11,round(a12)],[a13,a14,a15,round(a16),a17,round(a18)],
    [a19,a20,a21,round(a22),a23,round(a24)]
,[a25,a26,a27,round(a28),a29,round(a30)],[a31,a32,a33,round(a34),a35,round(a36)],
[a37,a38,a39,round(a40),a41,round(a42)],[a43,a44,a45,round(a46),a47,round(a48)],
[a49,a50,a51,round(a52),a53,round(a54)],[a55,a56,a57,round(a58),a59,round(a60)],
[a61,a62,a63,round(a64),a65,round(a66)],[a67,a68,a69,round(a70),a71,round(a72)],
[a73,a74,a75,round(a76),a77,round(a78)],[a79,a80,a81,round(a82),a83,round(a84)]]

vv1=np.mean([v1[0],v2[0],v3[0],v4[0],v5[0],v6[0],v7[0],v8[0],v9[0],v10[0]])
vv2=np.mean([v1[1],v2[1],v3[1],v4[1],v5[1],v6[1],v7[1],v8[1],v9[1],v10[1]])
vv3=np.mean([v1[2],v2[2],v3[2],v4[2],v5[2],v6[2],v7[2],v8[2],v9[2],v10[2]])
vv4=np.mean([v1[3],v2[3],v3[3],v4[3],v5[3],v6[3],v7[3],v8[3],v9[3],v10[3]])
vv5=np.mean([v1[4],v2[4],v3[4],v4[4],v5[4],v6[4],v7[4],v8[4],v9[4],v10[4]])
vv6=np.mean([v1[5],v2[5],v3[5],v4[5],v5[5],v6[5],v7[5],v8[5],v9[5],v10[5]])
table_111= pd.DataFrame(A,columns=['accuracy', 'f1_score', 'accuracy for unknown words',
'number of unknown words','accuracy for known words','number of known words']
,index=['A','B','C','D','E','F','G','H','I','J','K','L','M','.'])

#table_10= pd.DataFrame(A,
#columns=['accuracy', 'f1_score', 'accuracy for unknown words',
#         'number of unknown words','accuracy for known words','number of known words']
#,index=[list(tag2idx.keys())[0], list(tag2idx.keys())[1], list(tag2idx.keys())[2] , list(tag2idx.keys())[3] 
#, list(tag2idx.keys())[4] , list(tag2idx.keys())[5],list(tag2idx.keys())[6],list(tag2idx.keys())[7]
#,list(tag2idx.keys())[8],list(tag2idx.keys())[9],list(tag2idx.keys())[10],list(tag2idx.keys())[11],
#list(tag2idx.keys())[12],list(tag2idx.keys())[13]])

str_pythontex=[float("{0:.2f}".format(list(table_111.loc["A"])[0]*100)),float("{0:.2f}".format(list(table_111.loc["A"])[1]*100)),
float("{0:.2f}".format(list(table_111.loc["A"])[2]*100)),float("{0:.2f}".format(list(table_111.loc["A"])[4]*100)),
round(list(table_111.loc["A"])[3]),round(list(table_111.loc["A"])[5]),
float("{0:.2f}".format(list(table_111.loc["B"])[0]*100)),float("{0:.2f}".format(list(table_111.loc["B"])[1]*100)),
float("{0:.2f}".format(list(table_111.loc["B"])[2]*100)),float("{0:.2f}".format(list(table_111.loc["B"])[4]*100)),
round(list(table_111.loc["B"])[3]),round(list(table_111.loc["B"])[5]),
float("{0:.2f}".format(list(table_111.loc["C"])[0]*100)),float("{0:.2f}".format(list(table_111.loc["C"])[1]*100)),
float("{0:.2f}".format(list(table_111.loc["C"])[2]*100)),float("{0:.2f}".format(list(table_111.loc["C"])[4]*100)),
round(list(table_111.loc["C"])[3]),round(list(table_111.loc["C"])[5]),
float("{0:.2f}".format(list(table_111.loc["D"])[0]*100)),float("{0:.2f}".format(list(table_111.loc["D"])[1]*100)),
float("{0:.2f}".format(list(table_111.loc["D"])[2]*100)),float("{0:.2f}".format(list(table_111.loc["D"])[4]*100)),
round(list(table_111.loc["D"])[3]),round(list(table_111.loc["D"])[5]),
float("{0:.2f}".format(list(table_111.loc["E"])[0]*100)),float("{0:.2f}".format(list(table_111.loc["E"])[1]*100)),
float("{0:.2f}".format(list(table_111.loc["E"])[2]*100)),float("{0:.2f}".format(list(table_111.loc["E"])[4]*100)),
round(list(table_111.loc["E"])[3]),round(list(table_111.loc["E"])[5]),
float("{0:.2f}".format(list(table_111.loc["F"])[0]*100)),float("{0:.2f}".format(list(table_111.loc["F"])[1]*100)),
float("{0:.2f}".format(list(table_111.loc["F"])[2]*100)),float("{0:.2f}".format(list(table_111.loc["F"])[4]*100)),
round(list(table_111.loc["F"])[3]),round(list(table_111.loc["F"])[5]),
float("{0:.2f}".format(list(table_111.loc["G"])[0]*100)),float("{0:.2f}".format(list(table_111.loc["G"])[1]*100)),
float("{0:.2f}".format(list(table_111.loc["G"])[2]*100)),float("{0:.2f}".format(list(table_111.loc["G"])[4]*100)),
round(list(table_111.loc["G"])[3]),round(list(table_111.loc["G"])[5]),
float("{0:.2f}".format(list(table_111.loc["H"])[0]*100)),float("{0:.2f}".format(list(table_111.loc["H"])[1]*100)),
float("{0:.2f}".format(list(table_111.loc["H"])[2]*100)),float("{0:.2f}".format(list(table_111.loc["H"])[4]*100)),
round(list(table_111.loc["H"])[3]),round(list(table_111.loc["H"])[5]),
float("{0:.2f}".format(list(table_111.loc["I"])[0]*100)),float("{0:.2f}".format(list(table_111.loc["I"])[1]*100)),
float("{0:.2f}".format(list(table_111.loc["I"])[2]*100)),float("{0:.2f}".format(list(table_111.loc["I"])[4]*100)),
round(list(table_111.loc["I"])[3]),round(list(table_111.loc["I"])[5]),
float("{0:.2f}".format(list(table_111.loc["J"])[0]*100)),float("{0:.2f}".format(list(table_111.loc["J"])[1]*100)),
float("{0:.2f}".format(list(table_111.loc["J"])[2]*100)),float("{0:.2f}".format(list(table_111.loc["J"])[4]*100)),
round(list(table_111.loc["J"])[3]),round(list(table_111.loc["J"])[5]),
float("{0:.2f}".format(list(table_111.loc["K"])[0]*100)),float("{0:.2f}".format(list(table_111.loc["K"])[1]*100)),
float("{0:.2f}".format(list(table_111.loc["K"])[2]*100)),float("{0:.2f}".format(list(table_111.loc["K"])[4]*100)),
round(list(table_111.loc["K"])[3]),round(list(table_111.loc["K"])[5]),
float("{0:.2f}".format(list(table_111.loc["L"])[0]*100)),float("{0:.2f}".format(list(table_111.loc["L"])[1]*100)),
float("{0:.2f}".format(list(table_111.loc["L"])[2]*100)),float("{0:.2f}".format(list(table_111.loc["L"])[4]*100)),
round(list(table_111.loc["L"])[3]),round(list(table_111.loc["L"])[5]),
float("{0:.2f}".format(list(table_111.loc["M"])[0]*100)),float("{0:.2f}".format(list(table_111.loc["M"])[1]*100)),
float("{0:.2f}".format(list(table_111.loc["M"])[2]*100)),float("{0:.2f}".format(list(table_111.loc["M"])[4]*100)),
round(list(table_111.loc["M"])[3]),round(list(table_111.loc["M"])[5]),
float("{0:.2f}".format(list(table_111.loc["."])[0]*100)),float("{0:.2f}".format(list(table_111.loc["."])[1]*100)),
float("{0:.2f}".format(list(table_111.loc["."])[2]*100)),float("{0:.2f}".format(list(table_111.loc["."])[4]*100)),
round(list(table_111.loc["."])[3]),round(list(table_111.loc["."])[5]),float("{0:.2f}".format(vv1))
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

