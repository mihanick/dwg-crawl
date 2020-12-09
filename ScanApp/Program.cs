using DwgDump;
using DwgDump.Db;
using System;
//using System.Threading.Tasks;
using System.Diagnostics;
using System.IO;
using System.Threading.Tasks;

namespace DwgDump
{
	class Program
	{
		static void Main(string[] args)
		{
			string dir = @"C:\git\dwg-crawl\TestData";
			if (args.Length == 1)
				if (Directory.Exists(args[0]))
					dir = args[0];

			string dbName = "geometry";

			Scan(dir, dbName);
		}

		static void Scan(string dir, string dbName)
		{
			// Папка куда сохраняются все dwg
			string dataDir = @"C:\git\dwg-crawl\Data";

			//Открыть папку, выбрать все файлы двг из нее
			string[] dwgFiles = Directory.GetFiles(dir, "*.dwg", SearchOption.AllDirectories);
			Console.WriteLine("Db admin paswword please:");
			var password = Console.ReadLine();
			DbMongo db = new DbMongo(dbName, password);

			int numFiles2Process = 10;
			int n = 0;
			foreach (string dwgFile in dwgFiles)
			{
				CrawlDocument cDoc = new CrawlDocument(dwgFile);
				FileCopy(dwgFile, Path.Combine(dataDir, cDoc.FileId + ".dwg"));
				db.InsertIntoFiles(cDoc);
				n++;
				if (n > numFiles2Process)
					break;
			}

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

		/// <summary>
		/// Copies file overwriting and creating path if necessary
		/// </summary>
		/// <param name="sourceFullPath">Full source path</param>
		/// <param name="targetFullPath">Full destination path</param>
		[DebuggerStepThrough]
		private static void FileCopy(string sourceFullPath, string targetFullPath)
		{
			try
			{
				string fileName = Path.GetFileName(sourceFullPath);

				//http://msdn.microsoft.com/en-us/library/cc148994.aspx

				// Use Path class to manipulate file and directory paths. 
				string targetDirectory = Path.GetDirectoryName(targetFullPath);
				// To copy a folder's contents to a new location: 
				// Create a new target folder, if necessary. 
				if (!System.IO.Directory.Exists(targetDirectory))
				{
					System.IO.Directory.CreateDirectory(targetDirectory);
				}

				// To copy a file to another location and  
				// overwrite the destination file if it already exists.
				System.IO.File.Copy(sourceFullPath, targetFullPath, true);

				Console.WriteLine("Скопирован файл из {0} в {1}", sourceFullPath, targetFullPath);

			}
			catch (System.Exception e)
			{
				Console.WriteLine("[ОШИБКА] Ошибка копирования файла {0} в файл {1} '{2}'", sourceFullPath, targetFullPath, e.Message);
			}
		}


		/// <summary>
		/// Clears ReadOnly attribute from target File
		/// </summary>
		/// <param name="targetFilePath">Full Path to file</param>
		private static void ClearReadOnlyAttribute(string targetFilePath)
		{
			//https://msdn.microsoft.com/en-us/library/system.io.file.setattributes%28v=vs.110%29.aspx
			try
			{
				FileAttributes fileAttributes = File.GetAttributes(targetFilePath);
				fileAttributes &= ~FileAttributes.ReadOnly;
				File.SetAttributes(targetFilePath, fileAttributes);
			}
			catch (System.Exception e)
			{
				Console.WriteLine("[ОШИБКА] Ошибка задания атрибутов файла {0}: '{1}'", targetFilePath, e.Message);
			}
		}
	}
}
