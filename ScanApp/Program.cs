using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using Crawl;
using System.Threading.Tasks;
using System.Diagnostics;

namespace Crawl
{
    class Program
    {
        static void Main(string[] args)
        {
            //Открыть папку, выбрать все файлы двг из нее
            string dir = @"C:\Users\Mike Gladkikh\Documents\test 1800\AcDbLine";
            //string dir = @"\\FILESERVER\home\#АРХИВ 2014";

            string[] dwgFiles = Directory.GetFiles(dir, "*.dwg", SearchOption.TopDirectoryOnly);
            SqlDb sqlDB = new SqlDb();

            foreach (string dwgFile in dwgFiles)
            {
                CrawlDocument cDoc = new CrawlDocument(dwgFile);
                sqlDB.InsertIntoFiles(cDoc.Path, cDoc.docJson, cDoc.FileId, cDoc.Hash);
            }

            //Запуситить процессы по числу ядер процессоров каждый на своем ядре
            int numCores = 4;
            for (int i = 0; i < numCores; i++)
            {
                //crawlinNano();
                //http://cplus.about.com/od/learnc/a/multi-threading-using-task-parallel-library.htm
                Task.Factory.StartNew(() => crawlinNano());
            }

            //Процесс выбирает из базы случайным образом непросканированный файл и сканирует его в Json

            //Если файл изменился, то записывается его новый hash
        }

        static void crawlinNano()
        {
            ExecuteCommandLine(@"C:\Program Files (x86)\Nanosoft\nanoCAD СПДС Железобетон 2.4\nCad.exe");
        }

        static Process ExecuteCommandLine(string exePath, string arguments = "")
        {

            //http://stackoverflow.com/questions/206323/how-to-execute-command-line-in-c-get-std-out-results
            // Start the child process.
            Process p = new Process();
            // Redirect the output stream of the child process.
            //p.StartInfo.UseShellExecute = false;
            //p.StartInfo.RedirectStandardOutput = true;
            p.StartInfo.FileName = exePath;

            //http://social.msdn.microsoft.com/Forums/vstudio/en-US/1bbf1e4c-1911-40a4-861e-2b2990124314/execute-command-line-in-c
            p.StartInfo.Arguments = arguments;

            p.Start();
            return p;
        }
    }
}
