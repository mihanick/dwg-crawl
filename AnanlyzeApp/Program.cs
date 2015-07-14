using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

using Crawl;
// using System.Threading.Tasks;


namespace AnanlyzeApp
{
    class Program
    {
        static void Main(string[] args)
        {
            TextData();
        }

        static void ObjJsons()
        {   
            DbMongo sqlDB = new DbMongo();

            List<string> jsonObjs = sqlDB.GetObjectJsonByClassName("");

            string fileName = @"c:\Data\all.json";
           
            //https://msdn.microsoft.com/en-us/library/3aadshsx(v=vs.110).aspx
            FileStream fs = null;
            try
            {
                fs = new FileStream(fileName, FileMode.CreateNew);
                using (StreamWriter writer = new StreamWriter(fs, Encoding.Default))
                {
                    writer.WriteLine("{\"objects\":[");

                    foreach (string jsonObj in jsonObjs)
                        if (jsonObj != "")
                        {
                            writer.WriteLine(jsonObj);
                            writer.Write(",");
                        }
                    writer.WriteLine("]}");
                }
            }
            finally
            {
                if (fs != null)
                    fs.Dispose();
            }
             
        }

        static void TextData()
        {
            DbMongo sqlDB = new DbMongo();
            string className = "AcDbText";

            List<string> jsonObjs = sqlDB.GetObjectJsonByClassName(className);

            string fileName = @"c:\Data\" + className + ".csv";
            
            //https://msdn.microsoft.com/en-us/library/3aadshsx(v=vs.110).aspx
            FileStream fs = null;
            try
            {
                fs = new FileStream(fileName, FileMode.CreateNew);
                using (StreamWriter writer = new StreamWriter(fs, Encoding.Default))
                {
                    writer.WriteLine("Data" + "; " + "Value");

                    foreach (string jsonObj in jsonObjs)
                    {
                        crawlAcDbText cObject = jsonHelper.From<crawlAcDbText>(jsonObj);

                        writer.WriteLine(cObject.Position.ToString() + "; " + cObject.TextString);

                    }
                }
            }
            finally
            {
                if (fs != null)
                    fs.Dispose();
            }
              
        }

        static void LineData()
        {
            DbMongo sqlDB = new DbMongo();
            List<string> jsonOfLines = sqlDB.GetObjectJsonByClassName("AcDbLine");
            StreamWriter sw = new StreamWriter(@"C:\Data\LineData.csv");
            sw.WriteLine("Alignment" + "; " + "Length");

            foreach (string jsonLine in jsonOfLines)
            {
                crawlAcDbLine cLine = jsonHelper.From<crawlAcDbLine>(jsonLine);
                string rotated = "Rotated";
                if (cLine.StartPoint.X == cLine.EndPoint.X)
                    rotated = "Vertical";
                if (cLine.StartPoint.Y == cLine.EndPoint.Y)
                    rotated = "Horizontal";

                sw.WriteLine(rotated + "; " + cLine.Length);

            }

            sw.Close();
        }
    }
}
