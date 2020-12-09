using System;
//using System.Threading.Tasks;
using System.Diagnostics;
using System.IO;
using System.Threading.Tasks;

namespace DwgDump
{
	sealed class Program
	{
		static void Main(string[] args)
		{
			string dir = @"C:\git\dwg-crawl\TestData";

			if (args.Length == 1)
				if (Directory.Exists(args[0]))
					dir = args[0];

			DwgDump.Data.DirectoryScanner.Scan(dir);
			

			//Запуситить процессы по числу ядер процессоров каждый на своем ядре
			int numProcesses = 2;
			for (int i = 0; i < numProcesses; i++)
			{
				//crawlinNano();
				//http://cplus.about.com/od/learnc/a/multi-threading-using-task-parallel-library.htm
				Task.Factory.StartNew(() => CrawlinNano());
				//Процесс выбирает из базы случайным образом непросканированный файл и сканирует его в Json
				//Это пока выполняется ручным запуском нанокадов
				//Если файл изменился, то записывается его новый hash
			}


			Console.WriteLine("Wait for nano to close:");
			Console.ReadKey();
		}

		static void CrawlinNano()
		{
			ExecuteCommandLine(@"C:\Program Files\Nanosoft\nanoCAD x64 Plus 20.1\nCadS.exe", @" -b nSPDSComp -r SPDS -a nanoCAD_x64_SPDS_20.0");
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
			p.WaitForExit();
			return p;
		}
	}
}
