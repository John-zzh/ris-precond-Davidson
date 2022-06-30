#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Unit testing for Turbomole_class"""

import numpy as np
import os
import shutil
import unittest
import sys
from pathlib import Path
import mudslide
import yaml

from mudslide import TMModel
from mudslide.tracer import YAMLTrace

testdir = os.path.dirname(__file__)
print("issrizintest")
print(testdir)

@unittest.skipUnless(mudslide.turbomole_model.turbomole_is_installed(),
        "Turbomole must be installed")
class TestTMModel(unittest.TestCase):
    """Test Suite for TMModel class"""
    
    def setUp(self):
        self.turbomole_files = ["control"]
        for fl in self.turbomole_files:
            shutil.copy(testdir+"/turbomole_files/"+fl,".")

    def test_get_gs_ex_properties(self):
        """test for gs_ex_properties function"""
        model = TMModel(states = [0,1,2,3], turbomole_dir = ".") 
        positions = model.X 

        mom = [5.583286976987380000, -2.713959745507320000, 0.392059702162967000, 
                -0.832994241764031000, -0.600752326053757000, -0.384006560250834000, 
                -1.656414687719690000, 1.062437820195600000, -1.786171104341720000,
                -2.969087779972610000, 1.161804203506510000, -0.785009852486148000,
                2.145175145340160000, 0.594918215579156000, 1.075977514428970000,
                -2.269965412856570000,  0.495551832268249000,   1.487150300486560000]

        traj = mudslide.TrajectorySH(model, positions, mom, 3, tracer = YAMLTrace(name = "TMtrace"), dt = 20, max_time = 61, t0 = 1)
        results = traj.simulate()

        with open("TMtrace-0-log_0.yaml", "r") as f:
            data = yaml.safe_load(f) 
            
            gs_e_from_ridft_t1 = data[0]["electronics"]["hamiltonian"][0][0]
            ex_e_1_from_egrad_t1 = data[0]["electronics"]["hamiltonian"][1][1]
            ex_e_2_from_egrad_t1 = data[0]["electronics"]["hamiltonian"][2][2] 
            ex_e_3_from_egrad_t1 = data[0]["electronics"]["hamiltonian"][3][3] 


            gs_e_from_ridft_t21 = data[1]["electronics"]["hamiltonian"][0][0]
            ex_e_1_from_egrad_t21 = data[1]["electronics"]["hamiltonian"][1][1] 
            ex_e_2_from_egrad_t21 = data[1]["electronics"]["hamiltonian"][2][2] 
            ex_e_3_from_egrad_t21 = data[1]["electronics"]["hamiltonian"][3][3] 


            gs_e_from_ridft_t41   = data[2]["electronics"]["hamiltonian"][0][0]
            ex_e_1_from_egrad_t41 = data[2]["electronics"]["hamiltonian"][1][1] 
            ex_e_2_from_egrad_t41 = data[2]["electronics"]["hamiltonian"][2][2] 
            ex_e_3_from_egrad_t41 = data[2]["electronics"]["hamiltonian"][3][3] 

            gs_e_from_ridft_t61   = data[3]["electronics"]["hamiltonian"][0][0]
            ex_e_1_from_egrad_t61 = data[3]["electronics"]["hamiltonian"][1][1] 
            ex_e_2_from_egrad_t61 = data[3]["electronics"]["hamiltonian"][2][2] 
            ex_e_3_from_egrad_t61 = data[3]["electronics"]["hamiltonian"][3][3] 

            dm_from_mudslide_t1 = data[0]["density_matrix"]
            dm_from_mudslide_t21 = data[1]["density_matrix"]
            dm_from_mudslide_t41 = data[2]["density_matrix"]
            dm_from_mudslide_t61 = data[3]["density_matrix"]

            gs_grad_from_rdgrad_t1 = data[0]["electronics"]["force"][0] 
            ex_st_1_gradients_from_egrad_t1 = data[0]["electronics"]["force"][1]
            ex_st_2_gradients_from_egrad_t1 = data[0]["electronics"]["force"][2]
            ex_st_3_gradients_from_egrad_t1 = data[0]["electronics"]["force"][3]

            gs_grad_from_rdgrad_t21 = data[1]["electronics"]["force"][0] 
            ex_st_1_gradients_from_egrad_t21 = data[1]["electronics"]["force"][1]
            ex_st_2_gradients_from_egrad_t21 = data[1]["electronics"]["force"][2]
            ex_st_3_gradients_from_egrad_t21 = data[1]["electronics"]["force"][3]


            gs_grad_from_rdgrad_t41 = data[2]["electronics"]["force"][0] 
            ex_st_1_gradients_from_egrad_t41 = data[2]["electronics"]["force"][1]
            ex_st_2_gradients_from_egrad_t41 = data[2]["electronics"]["force"][2]
            ex_st_3_gradients_from_egrad_t41 = data[2]["electronics"]["force"][3]


            gs_grad_from_rdgrad_t61          = data[3]["electronics"]["force"][0] 
            ex_st_1_gradients_from_egrad_t61 = data[3]["electronics"]["force"][1]
            ex_st_2_gradients_from_egrad_t61 = data[3]["electronics"]["force"][2]
            ex_st_3_gradients_from_egrad_t61 = data[3]["electronics"]["force"][3]


            derivative_coupling01_from_egrad_t1 = data[0]["electronics"]["derivative_coupling"][1][0]

            derivative_coupling02_from_egrad_t21 = data[1]["electronics"]["derivative_coupling"][1][0]
            derivative_coupling02_from_egrad_t41 = data[2]["electronics"]["derivative_coupling"][1][0]
            derivative_coupling02_from_egrad_t61 = data[3]["electronics"]["derivative_coupling"][1][0]

            coord_t1 = data[0]["position"]
            coord_t21 = data[1]["position"]
            coord_t41 = data[2]["position"]
            coord_t61 = data[3]["position"]


            mom_t1 = data[0]["momentum"]
            mom_t21 = data[1]["momentum"]
            mom_t41 = data[2]["momentum"]
            mom_t61 = data[3]["momentum"]




        gs_energy_ref_t1    = -78.40037178008
        excited_1_energy_ref_t1 = -78.10536751498037
        excited_2_energy_ref_t1 = -78.08798681826964
        excited_3_energy_ref_t1 = -78.07233323013524


        gs_energy_ref_t21 =   -78.40048983202 
        excited_1_energy_ref_t21 = -78.10532062540331 
        excited_2_energy_ref_t21 =  -78.09028427832553 
        excited_3_energy_ref_t21 =  -78.07386514254479

        gs_energy_ref_t41 =  -78.39991626903
        excited_1_energy_ref_t41 = -78.10576552630319 
        excited_2_energy_ref_t41 = -78.0931135494284  
        excited_3_energy_ref_t41 = -78.07611866536453  

        gs_energy_ref_t61 =  -78.39828221685
        excited_1_energy_ref_t61 = -78.10617439312973 
        excited_2_energy_ref_t61 = -78.09583148620752 
        excited_3_energy_ref_t61 = -78.07876209169358  

        dm_t_1_ref = np.array([ [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0] ])
        
        dm_t_21_ref = np.array(
                             [[1.749918392647146e-10, 0.0, -1.7139005307393475e-08, -1.3008664617923628e-07, -3.585479628707707e-08, 
                             1.1438381702129466e-06, -2.5133448026917437e-06, 1.2936308187802462e-05], [-1.7139005307393475e-08, 
                             1.3008664617923628e-07, 9.838333655688461e-05, 0.0, -0.0008468026647047002, -0.00013868348815851338, 
                            -0.009370521065837013, -0.0031353922163937214], [-3.585479628707707e-08, -1.1438381702129466e-06, 
                            -0.0008468026647047002, 0.00013868348815851338, 0.007484070866137629, 0.0, 0.08507344465490325, 0.013777962650003415],
                             [-2.5133448026917437e-06, -1.2936308187802462e-05, -0.009370521065837013, 0.0031353922163937214, 0.08507344465490325, 
                            -0.013777962650003415, 0.9924175456223143, 0.0]])


        dm_t_41_ref = np.array(
                            [[4.826545363136371e-10, 2.1002632740604218e-26, -8.642983561343313e-08, -5.475746492935279e-07, 
                            -1.534857123911518e-07, 3.0809108101072967e-06, -8.798081399856754e-06, 1.9885306695281293e-05], 
                            [-8.642983561343317e-08, 5.47574649293528e-07, 0.0006367040811016045, 5.421010862427522e-20, 
                            -0.003467827991925316, -0.0007258348852723232, -0.02098451863317442, -0.013542377899730226], 
                            [-1.5348571239115213e-07, -3.080910810107297e-06, -0.003467827991925316, 0.0007258348852723232, 
                            0.019715072723486934, 0.0, 0.12973095959971473, 0.04983687465676305], 
                            [-8.798081399856754e-06, -1.9885306695281293e-05, -0.020984518633174416, 
                            0.013542377899730222, 0.12973095959971473, -0.04983687465676306, 0.979648222712757, 0.0]])




        dm_t_61_ref = np.array(
                            [[6.432275475720007e-10, 1.0339757656912846e-25, -1.036898669113144e-07, -1.1221892741822371e-06, 
                            -2.5376576537302465e-07, 4.039679485408739e-06, -1.6058030190203505e-05, 1.917586362604496e-05], 
                            [-1.0368986691131464e-07, 1.1221892741822375e-06, 0.001974511757750202, 2.168404344971009e-19, 
                            -0.00700680819444698, -0.0010939317678770434, -0.030866049729977574, -0.031106397831935768], 
                            [-2.5376576537302465e-07, -4.03967948540874e-06, -0.00700680819444698, 0.0010939317678770439,
                             0.025470624618535425, -1.734723475976807e-18, 0.12676590350415978, 0.09328443356038119], 
                            [-1.6058030190203508e-05, -1.9175863626044968e-05, -0.03086604972997757, 0.031106397831935764, 
                            0.12676590350415978, -0.0932844335603812, 0.9725548629804863, 0.0]])


        gs_gradients_t1_ref = np.array(
                                [-2.929559e-10, -0.0, 0.01142713, -2.836042e-10, -0.0, -0.01142713, 0.01198766, -5.320008e-14, 
                                0.007575883, -0.01198766, 5.422816e-14, 0.007575883, 0.01198766, -5.395386e-14, -0.007575883, 
                               -0.01198766, 5.336176e-14, -0.007575883]) 


        ex_st_1_gradients_t1_ref = np.array([
                                -2.557566e-10, -0.0, 0.01232717, -2.503219e-10, -0.0, -0.01232717, 0.02187082, -1.264907e-13, 
                                 0.03277518, -0.02187082, 1.276739e-13, 0.03277518, 0.02187082, -1.271331e-13, -0.03277518, 
                                -0.02187082, 1.262567e-13, -0.03277518]) 

        ex_st_2_gradients_t1_ref = np.array([
                                -2.974175e-10, -0.0, 0.08614947, -2.8399e-10, -0.0, -0.08614947, 0.03613404, -1.676159e-14, 
                                 0.009036918, -0.03613404, 1.673288e-14, 0.009036918, 0.03613404, -1.644249e-14, -0.009036918, 
                                -0.03613404, 1.6608e-14, -0.009036918]) 

        ex_st_3_gradients_t1_ref = np.array([
                                -3.02627e-10, -0.0, 0.02062589, -2.883125e-10, -0.0, -0.02062589, 0.03944075, -2.061206e-14, 
                                 0.0259814, -0.03944075, 2.081753e-14, 0.0259814, 0.03944075, -1.839483e-14, -0.0259814, 
                                -0.03944075, 1.834661e-14, -0.0259814])



        gs_gradients_t21_ref = np.array(
                                [-0.01882071, 0.0008168962, 0.01070758, 0.001060233, 0.0001804384, -0.01568363, 0.01964265, 
                                -0.0002368164, 0.0134072, -0.001273064, -0.0004099903, 0.002688204, 0.004777519, -0.0001934963, 
                                -0.00541991, -0.005386627, -0.0001570316, -0.005699438])


        ex_st_1_gradients_t21_ref = np.array([
                                -0.008393041, -0.001267764, 0.01299881, -0.004781268, -0.0002859212, -0.01815933, 0.02713894, 
                                 0.0005779374, 0.03850215, -0.0147826, 0.0005513749, 0.0265579, 0.01647766, 0.0002578082, 
                                -0.02925562, -0.01565969, 0.0001665643, -0.03064392]) 

        ex_st_2_gradients_t21_ref = np.array([
                                -0.00830215, 0.0006506364, 0.08606266, 0.01015423, 5.456523e-05, -0.09005891, 0.03879733, 
                                -0.0002625103, 0.0129691, -0.03045149, -0.0002571386, 0.005181622, 0.02646374, -8.047436e-05, 
                                -0.00548443, -0.03666166, -0.0001050784, -0.008670045])

        ex_st_3_gradients_t21_ref = np.array([
                                -0.02244935, 0.0004358261, 0.02192937, -0.01354063, -0.0004081886, -0.02578153, 0.04657908, 
                                -0.0002803852, 0.02873018, -0.02771836, 0.0001022478, 0.02266842, 0.0417367, 0.0003729335,
                                -0.02862104, -0.02460743, -0.0002224337, -0.0189254])





        gs_gradients_t41_ref = np.array(
                                [-0.03491684, 0.001694538, 0.0118373, 0.003215844, 0.0004426022, -0.02153318, 0.02346781, 
                                -0.0003911119, 0.01691057, 0.01062317, -0.0009627295, -0.002617742, -0.005290681, -0.0004332228, 
                                -0.001634175, 0.002900698, -0.0003500765, -0.00296277])

        ex_st_1_gradients_t41_ref = np.array([
                                -0.01430738, -0.002384127, 0.01649185, -0.008362121, -0.0004703724, -0.02684268, 0.02877637, 
                                 0.001214675, 0.04157716, -0.006528649, 0.0009058438, 0.01970963, 0.00809321, 0.0004171751, 
                                -0.02366734, -0.00767143, 0.0003168054, -0.02726862])


        ex_st_2_gradients_t41_ref = np.array([
                                -0.01674236, 0.00133935, 0.08678011, 0.01853151, 0.0001923298, -0.0952162, 0.03981963, 
                                -0.0005061157, 0.01549563, -0.02277533, -0.0005763445, 0.0007327416, 0.01523734, -0.0002129648, 
                                -0.0007294144, -0.03407079, -0.0002362545, -0.007062879]) 

        ex_st_3_gradients_t41_ref = np.array([
                                -0.03955009, 0.0009339586, 0.0258132, -0.02342849, -0.0007902726, -0.03258427, 0.04821983, 
                                -0.0005407365, 0.02906081, -0.01571054, 0.0001230272, 0.01922434, 0.04058162, 0.0008309709, 
                                -0.02954673, -0.01011233, -0.0005569476, -0.01196736]) 

        gs_gradients_t61_ref = np.array(
                                [-0.04618272, 0.002698636, 0.014452, 0.006951907, 0.0008067209, -0.0275698, 0.02278034, 
                                -0.0005565761, 0.01764507, 0.02233488, -0.001634679, -0.007655337, -0.01735295, -0.0007404526, 
                                 0.003295346, 0.01146855, -0.0005736495, -0.0001672729])


        ex_st_1_gradients_t61_ref = np.array([
                                -0.01640374, -0.003209886, 0.02223014, -0.009747237, -0.0005206459, -0.03664599, 0.02629475, 
                                0.001784072, 0.04150362, 0.001696873, 0.00105478, 0.01311563, -0.00251063, 0.0004440675, 
                                -0.0166874, 0.000669982, 0.0004476123, -0.02351599]) 


        ex_st_2_gradients_t61_ref = np.array([
                                -0.02353193, 0.002107828, 0.08906172, 0.02479332, 0.0004482579, -0.1008459, 0.03812769, -0.0007738158, 
                                 0.01574817, -0.01379321, -0.0009688363, -0.003936621, 0.003777257, -0.0004000967, 0.004732261, 
                                -0.02937312, -0.0004133372, -0.004759608]) 


        ex_st_3_gradients_t61_ref = np.array([
                                -0.04900744, 0.001530575, 0.03072285, -0.02857704, -0.001107381, -0.03921126, 0.04414275, 
                                -0.0008437878, 0.02697915, -0.005331454, 7.852523e-05, 0.01655062, 0.03628921, 0.001273792, 
                                -0.02906087, 0.002483974, -0.0009317237, -0.005980494]) 


        derivative_coupling01_t1_ref = np.array([-0.0, -6.957965e-11, -0.0, -0.0, -8.936381e-11, -0.0, -0.0, 0.1983295, 
                                             -0.0, -0.0, -0.1983295, -0.0, -0.0, 0.1983295, -0.0, -0.0, -0.1983295, -0.0])


        derivative_coupling02_t21_ref = np.array( [0.001653841, -0.01234554, 0.0001328363, 0.0007634082, 0.009828648, 
                                            0.0001557527, -0.0005532372, 0.1940909, -0.001014699, -0.0007221076, -0.1989962, 
                                            0.0009025694, -0.0004040651, 0.1972493, 0.0003609095, -0.0002632777, -0.1933652, -0.0004283189])

        derivative_coupling02_t41_ref = np.array([0.003300368, -0.0241384, 0.0002022263, 0.001491023, 0.0204431, 0.0003815128, -0.001067507, 
                                                0.1882349, -0.002013107, -0.001441587, -0.198791, 0.00170646, -0.0007851564, 0.1946224, 0.0006793311, 
                                                 -0.0005304136, -0.187576, -0.0008375983]) 

        derivative_coupling02_t61_ref = np.array( [0.004883346, -0.03421491, 0.0001090853, 0.002146812, 0.0313972, 
                                                 0.0007299921, -0.001488015, 0.1809848, -0.002930583, -0.00210204, -0.1977834, 
                                                 0.002402343, -0.001131869, 0.1904037, 0.0009417586, -0.0007929434, -0.1814259, -0.001225963])

        coord_t1_ref = np.array([     
                                                0.00000000000000,      0.00000000000000,      1.24876020687021, 
                                                0.00000000000000,      0.00000000000000,     -1.24876020687021, 
                                                1.74402803906479,      0.00000000000000,      2.32753092373037,      
                                               -1.74402803906479,      0.00000000000000,      2.32753092373037, 
                                                1.74402803906479,      0.00000000000000,     -2.32753092373037, 
                                               -1.74402803906479,      0.00000000000000,     -2.32753092373037 
                                                ])

        coord_t21_ref = np.array([ 
                                                0.005104798434123317, -0.002481372984005959, 1.24930724947749, 
                                               -0.0007616065116373071, -0.0005492677606645478, -1.2492998864910603, 
                                                1.7302893093023415, 0.011566135473002911, 2.3109143719534857, 
                                               -1.780644433296515, 0.01264787882682078, 2.3218134254784752, 
                                                1.771674981184649, 0.006476524598378913, -2.318645831731036, 
                                               -1.7730334994079007, 0.005394781244569587, -2.3141696355189327
                                                ])

        coord_t41_ref = np.array([
                                                0.009799088122192396, -0.004954776453268342, 1.2502552924781203, 
                                               -0.0017708169424118887, -0.0011059996568314437, -1.2503110070913446, 
                                                1.7266921603316754, 0.023071223171798605, 2.3005531915070883, 
                                               -1.8232958970778255, 0.025318019886482614, 2.321031482262276, 
                                                1.8084091812544802, 0.013034247348808321, -2.3159923482028195, 
                                               -1.8073966920081348, 0.010741132392637374, -2.3049289412118146
                                                ])



        coord_t61_ref = np.array([
                                                0.013770164984400942, -0.0074111015584020275, 1.2516753555757087, 
                                               -0.003208440670312855, -0.0016771824753665605, -1.2519179635540834, 
                                                1.73359382975139, 0.034458577267830665, 2.2965193698769157, 
                                               -1.8693679889757182, 0.03801494744103415, 2.3244352206284242, 
                                                1.8539791457466963, 0.019772895925122873, -2.319772021993897, 
                                               -1.8439616243526524, 0.01596622032378054, -2.2982938790533853
                                                ])

        mom_t1_ref = np.array([                  5.58328697698738, -2.71395974550732, 0.3920597021629669, -0.8329942417640311, 
                                                 -0.600752326053757, -0.384006560250834, -1.65641468771969, 1.0624378201956, 
                                                 -1.78617110434172, -2.96908777997261, 1.16180420350651, -0.785009852486148, 
                                                  2.14517514534016, 0.594918215579156, 1.07597751442897, -2.26996541285657, 
                                                  0.495551832268249, 1.48715030048656])


        mom_t21_ref = np.array([
                                                 5.35879347396111, -2.70960148450732, 0.817612302162967, -0.9684005446471562, 
                                                -0.604834212053757, -0.8480807602508341, -0.7962163877196898, 1.0596339681953941, 
                                                -1.23905530434172, -3.6406788799726097, 1.162826681506718, -0.2985116524861479, 
                                                 2.95694964534016, 0.598647550578972, 0.5299531144289701, -2.9104472128565697, 
                                                 0.4933274952684325, 1.0380823004865598])


        mom_t41_ref = np.array([   
                                                 4.738799073961109, -2.6959036375073206, 1.295038002162967, -1.338091744647156, 
                                                 -0.616818824053757, -1.4317387602508342, 0.15177271228031022, 1.051422751195394, 
                                                 -0.6611454043417201, -4.07496787997261, 1.165079431506718, 0.12041594751385207, 
                                                  3.78013284534016, 0.610686594578972, -0.05172458557102991, -3.25764481285657, 
                                                  0.4855336822684325, 0.72915470048656])


        mom_t61_ref = np.array([   
                                                 3.8532237739611097, -2.6712583015073204, 1.8603985021629672, -1.8581470446471562, 
                                                 -0.6357953600537569, -2.149694060250834, 1.0753985122803102, 1.0375775081953942, 
                                                 -0.1007458043417201, -4.28538781997261, 1.167094955806718, 0.478165547513852, 
                                                  4.54884114534016, 0.6317342235789719, -0.6378005855710299, -3.33392837285657, 
                                                  0.47064696926843247, 0.5496761604865599])

        self.assertAlmostEqual(gs_energy_ref_t1, gs_e_from_ridft_t1, places=8)
        self.assertAlmostEqual(excited_1_energy_ref_t1, ex_e_1_from_egrad_t1, places=8)
        self.assertAlmostEqual(excited_2_energy_ref_t1, ex_e_2_from_egrad_t1, places=8)
        self.assertAlmostEqual(excited_3_energy_ref_t1, ex_e_3_from_egrad_t1, places=8)


        self.assertAlmostEqual(gs_energy_ref_t21, gs_e_from_ridft_t21, places=8)
        self.assertAlmostEqual(excited_1_energy_ref_t21, ex_e_1_from_egrad_t21, places=8)
        self.assertAlmostEqual(excited_2_energy_ref_t21, ex_e_2_from_egrad_t21, places=8)
        self.assertAlmostEqual(excited_3_energy_ref_t21, ex_e_3_from_egrad_t21, places=8)

        self.assertAlmostEqual(gs_energy_ref_t41, gs_e_from_ridft_t41, places=8)
        self.assertAlmostEqual(excited_1_energy_ref_t41, ex_e_1_from_egrad_t41, places=8)
        self.assertAlmostEqual(excited_2_energy_ref_t41, ex_e_2_from_egrad_t41, places=8)
        self.assertAlmostEqual(excited_3_energy_ref_t41, ex_e_3_from_egrad_t41, places=8)

        self.assertAlmostEqual(gs_energy_ref_t61, gs_e_from_ridft_t61, places=8)
        self.assertAlmostEqual(excited_1_energy_ref_t61, ex_e_1_from_egrad_t61, places=8)
        self.assertAlmostEqual(excited_2_energy_ref_t61, ex_e_2_from_egrad_t61, places=8)
        self.assertAlmostEqual(excited_3_energy_ref_t61, ex_e_3_from_egrad_t61, places=8)

        np.testing.assert_almost_equal(dm_t_1_ref, dm_from_mudslide_t1,  decimal = 8)
        np.testing.assert_almost_equal(dm_t_21_ref, dm_from_mudslide_t21,  decimal = 8)
        np.testing.assert_almost_equal(dm_t_41_ref, dm_from_mudslide_t41,  decimal = 8)
        np.testing.assert_almost_equal(dm_t_61_ref, dm_from_mudslide_t61,  decimal = 8)

        np.testing.assert_almost_equal(gs_gradients_t1_ref, gs_grad_from_rdgrad_t1,decimal = 8)
        np.testing.assert_almost_equal(ex_st_1_gradients_t1_ref, ex_st_1_gradients_from_egrad_t1,decimal = 8)
        np.testing.assert_almost_equal(ex_st_2_gradients_t1_ref, ex_st_2_gradients_from_egrad_t1,decimal = 8)
        np.testing.assert_almost_equal(ex_st_3_gradients_t1_ref, ex_st_3_gradients_from_egrad_t1,decimal = 8)

        np.testing.assert_almost_equal(gs_gradients_t21_ref, gs_grad_from_rdgrad_t21,decimal = 8)
        np.testing.assert_almost_equal(ex_st_1_gradients_t21_ref, ex_st_1_gradients_from_egrad_t21,decimal = 8)
        np.testing.assert_almost_equal(ex_st_2_gradients_t21_ref, ex_st_2_gradients_from_egrad_t21,decimal = 8)
        np.testing.assert_almost_equal(ex_st_3_gradients_t21_ref, ex_st_3_gradients_from_egrad_t21,decimal = 8)

        np.testing.assert_almost_equal(gs_gradients_t41_ref, gs_grad_from_rdgrad_t41,decimal = 8)
        np.testing.assert_almost_equal(ex_st_1_gradients_t41_ref, ex_st_1_gradients_from_egrad_t41,decimal = 8)
        np.testing.assert_almost_equal(ex_st_2_gradients_t41_ref, ex_st_2_gradients_from_egrad_t41,decimal = 8)
        np.testing.assert_almost_equal(ex_st_3_gradients_t41_ref, ex_st_3_gradients_from_egrad_t41,decimal = 8)

        np.testing.assert_almost_equal(gs_gradients_t61_ref, gs_grad_from_rdgrad_t61,decimal = 8)
        np.testing.assert_almost_equal(ex_st_1_gradients_t61_ref, ex_st_1_gradients_from_egrad_t61,decimal = 8)
        np.testing.assert_almost_equal(ex_st_2_gradients_t61_ref, ex_st_2_gradients_from_egrad_t61,decimal = 8)
        np.testing.assert_almost_equal(ex_st_3_gradients_t61_ref, ex_st_3_gradients_from_egrad_t61,decimal = 8)

        np.testing.assert_almost_equal(derivative_coupling01_t1_ref, derivative_coupling01_from_egrad_t1, decimal = 8)
        np.testing.assert_almost_equal(derivative_coupling02_t21_ref, derivative_coupling02_from_egrad_t21, decimal = 8)
        np.testing.assert_almost_equal(derivative_coupling02_t41_ref, derivative_coupling02_from_egrad_t41, decimal = 8)
        np.testing.assert_almost_equal(derivative_coupling02_t61_ref, derivative_coupling02_from_egrad_t61, decimal = 8)

        np.testing.assert_almost_equal(coord_t1_ref, coord_t1, decimal = 8)
        np.testing.assert_almost_equal(coord_t21_ref, coord_t21, decimal = 8)
        np.testing.assert_almost_equal(coord_t41_ref, coord_t41, decimal = 8)
        np.testing.assert_almost_equal(coord_t61_ref, coord_t61, decimal = 8)

        np.testing.assert_almost_equal(mom_t1_ref, mom_t1, decimal = 8)
        np.testing.assert_almost_equal(mom_t21_ref, mom_t21, decimal = 8)
        np.testing.assert_almost_equal(mom_t41_ref, mom_t41, decimal = 8)
        np.testing.assert_almost_equal(mom_t61_ref, mom_t61, decimal = 8)
        np.testing.assert_almost_equal(dm_t_1_ref, dm_from_mudslide_t1,  decimal = 8)
        np.testing.assert_almost_equal(dm_t_21_ref, dm_from_mudslide_t21,  decimal = 8)
        np.testing.assert_almost_equal(dm_t_41_ref, dm_from_mudslide_t41,  decimal = 8)
        np.testing.assert_almost_equal(dm_t_61_ref, dm_from_mudslide_t61,  decimal = 8)

        np.testing.assert_almost_equal(gs_gradients_t1_ref, gs_grad_from_rdgrad_t1,decimal = 8)
        np.testing.assert_almost_equal(ex_st_1_gradients_t1_ref, ex_st_1_gradients_from_egrad_t1,decimal = 8)
        np.testing.assert_almost_equal(ex_st_2_gradients_t1_ref, ex_st_2_gradients_from_egrad_t1,decimal = 8)
        np.testing.assert_almost_equal(ex_st_3_gradients_t1_ref, ex_st_3_gradients_from_egrad_t1,decimal = 8)

        np.testing.assert_almost_equal(gs_gradients_t21_ref, gs_grad_from_rdgrad_t21,decimal = 8)
        np.testing.assert_almost_equal(ex_st_1_gradients_t21_ref, ex_st_1_gradients_from_egrad_t21,decimal = 8)
        np.testing.assert_almost_equal(ex_st_2_gradients_t21_ref, ex_st_2_gradients_from_egrad_t21,decimal = 8)
        np.testing.assert_almost_equal(ex_st_3_gradients_t21_ref, ex_st_3_gradients_from_egrad_t21,decimal = 8)

        np.testing.assert_almost_equal(gs_gradients_t41_ref, gs_grad_from_rdgrad_t41,decimal = 8)
        np.testing.assert_almost_equal(ex_st_1_gradients_t41_ref, ex_st_1_gradients_from_egrad_t41,decimal = 8)
        np.testing.assert_almost_equal(ex_st_2_gradients_t41_ref, ex_st_2_gradients_from_egrad_t41,decimal = 8)
        np.testing.assert_almost_equal(ex_st_3_gradients_t41_ref, ex_st_3_gradients_from_egrad_t41,decimal = 8)

        np.testing.assert_almost_equal(gs_gradients_t61_ref, gs_grad_from_rdgrad_t61,decimal = 8)
        np.testing.assert_almost_equal(ex_st_1_gradients_t61_ref, ex_st_1_gradients_from_egrad_t61,decimal = 8)
        np.testing.assert_almost_equal(ex_st_2_gradients_t61_ref, ex_st_2_gradients_from_egrad_t61,decimal = 8)
        np.testing.assert_almost_equal(ex_st_3_gradients_t61_ref, ex_st_3_gradients_from_egrad_t61,decimal = 8)

        np.testing.assert_almost_equal(derivative_coupling01_t1_ref, derivative_coupling01_from_egrad_t1, decimal = 8)
        np.testing.assert_almost_equal(derivative_coupling02_t21_ref, derivative_coupling02_from_egrad_t21, decimal = 8)
        np.testing.assert_almost_equal(derivative_coupling02_t41_ref, derivative_coupling02_from_egrad_t41, decimal = 8)
        np.testing.assert_almost_equal(derivative_coupling02_t61_ref, derivative_coupling02_from_egrad_t61, decimal = 8)

        np.testing.assert_almost_equal(coord_t1_ref, coord_t1, decimal = 8)
        np.testing.assert_almost_equal(coord_t21_ref, coord_t21, decimal = 8)
        np.testing.assert_almost_equal(coord_t41_ref, coord_t41, decimal = 8)
        np.testing.assert_almost_equal(coord_t61_ref, coord_t61, decimal = 8)

    def tearDown(self):
        turbomole_files = ["TMtrace-0.yaml", "dipl_a", "ciss_a", "TMtrace-0-log_0.yaml", "TMtrace-0-events.yaml", "egradmonlog.1",  "excitationlog.1" ]
        for f in turbomole_files:
            os.remove(f)

if __name__ == '__main__':
    unittest.main()
