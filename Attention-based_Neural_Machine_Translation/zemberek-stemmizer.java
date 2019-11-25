import zemberek.morphology.TurkishMorphology;
import zemberek.morphology.analysis.SingleAnalysis;
import zemberek.morphology.analysis.WordAnalysis;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;

public class StemmingAndLemmatization {

    public static void main(String[] args) throws IOException {
        TurkishMorphology morphology = TurkishMorphology.createWithDefaults();
        File file = new File("tr.dev"); // for all files
        Scanner input = new Scanner(file);
        BufferedWriter bw = new BufferedWriter(new FileWriter(new File("tr.dev_stemmized")));

        while (input.hasNext()) {
            String[] sentence = input.nextLine().split("\\s+");//.split(" ");
            for (int i = 0; i < sentence.length; i++) {
                String word=sentence[i];
                WordAnalysis results = morphology.analyze(word);
                //System.out.println(word);
                for (SingleAnalysis result : results) {
                    //System.out.println(result.formatLong());
                    //System.out.println("\tStems = " + result.getStemAndEnding().toString().replaceAll("-"," "));
                    bw.append(result.getStemAndEnding().toString().replaceAll("-"," ")+" ");
                    break;
                    // System.out.println("\tLemmas = " + result.getLemmas());
                }
            }
            bw.newLine();
            //System.out.println(sentence.toString()+"fdsafad");

        }
        bw.close();
    }
}
