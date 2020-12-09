using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DwgDump.Data
{
	public static class DirectoryScanner
	{
		public static void Scan(string dir)
		{
			//Открыть папку, выбрать все файлы двг из нее
			string[] dwgFiles = Directory.GetFiles(dir, "*.dwg", SearchOption.AllDirectories);
			DbMongo db = DbMongo.Instance;

			int numFiles2Process = 10;
			int n = 0;
			foreach (string dwgFile in dwgFiles)
			{
				CrawlDocument cDoc = new CrawlDocument(dwgFile);
				FileCopy(dwgFile, Path.Combine(db.DataDir, cDoc.FileId + ".dwg"));
				db.InsertIntoFiles(cDoc);
				n++;
				if (n > numFiles2Process)
					break;
			}
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

				ClearReadOnlyAttribute(targetFullPath);
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
