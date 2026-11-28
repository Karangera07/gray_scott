import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.*;
import java.nio.file.*;
/** Reads jobs.txt lines: bc|Dexpr|F|k|outfile  (bc = noflux or dirichlet). Skips jobs whose outfile exists. */
public class Generate {
  public static void main(String[] args) throws Exception {
    String dir = "C:/Users/Karan/AppData/Local/Temp/claude/2532819e9fab9533/2f06b4ff-306e-4f8c-b6ca-78f026e66175/scratchpad/comsol";
    Model m = ModelUtil.load("M", dir + "/model.mph");
    PrintWriter log = new PrintWriter(new FileWriter(dir + "/generate.log", true), true);
    for (String line : Files.readAllLines(Paths.get(dir + "/jobs.txt"))) {
      line = line.trim(); if (line.isEmpty() || line.startsWith("#")) continue;
      String[] f = line.split(java.util.regex.Pattern.quote("|"));
      String bc = f[0], dexpr = f[1], F = f[2], k = f[3], out = f[4];
      if (new File(out).exists()) { log.println("skip " + out); continue; }
      new File(out).getParentFile().mkdirs();
      long t0 = System.currentTimeMillis();
      m.param().set("feed", F); m.param().set("kill", k);
      m.component("comp1").physics("c").feature("cfeq1").set("c", new String[]{dexpr});
      boolean dir1 = bc.equals("dirichlet");
      for (String ph : new String[]{"c", "c2"}) {
        m.component("comp1").physics(ph).feature("dir1").active(dir1);
        m.component("comp1").physics(ph).feature("dir1").set("r", new String[]{"0"});
      }
      log.println("run " + line); 
      try {
        m.study("std1").run();
        String tmp = out + ".part";
        m.result().export("data1").set("filename", tmp);
        m.result().export("data1").run();
        Files.move(Paths.get(tmp), Paths.get(out), StandardCopyOption.REPLACE_EXISTING);
        log.println("  done in " + (System.currentTimeMillis() - t0) / 1000 + " s -> " + out);
      } catch (Exception e) { log.println("  FAILED: " + e.getMessage()); }
    }
    log.println("ALL DONE");
  }
}
