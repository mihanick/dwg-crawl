using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Crawl;
using System.IO;

namespace AnanlyzeApp
{
    class Program
    {
        static void Main(string[] args)
        {
            //LineData();
            TextData();
        }
        static void TextData()
        {
            SqlDb sqlDB = new SqlDb();
            string className = "AcDbText";

            List<string> jsonObjs = sqlDB.GetObjectJsonByClassName(className);

            string fileName = @"F:\Data\" + className + ".csv";

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
            SqlDb sqlDB = new SqlDb();
            List<string> jsonOfLines = sqlDB.GetObjectJsonByClassName("AcDbLine");
            StreamWriter sw = new StreamWriter(@"F:\Data\LineData.csv");
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
