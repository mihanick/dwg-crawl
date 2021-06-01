using DwgDump.Data;
using DwgDump.Enitites;
using Multicad;
using Multicad.DatabaseServices;
using Multicad.Runtime;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;

namespace DwgDump
{
	public static class Main
	{
		static CrawlAcDbDocument currentDocument = null;

		[CommandMethod("sfd", CommandFlags.NoCheck | CommandFlags.NoPrefix)]
		public static void ScanFolder()
		{
			// DirectoryScanner.Scan(@"C:\Users\mihan\Desktop\DwgCrawlTest");
			DirectoryScanner.Scan(@"C:\py\Data");
		}

		/// <summary>
		/// Opens unscanned document
		/// </summary>
		[CommandMethod("grd", CommandFlags.NoCheck | CommandFlags.NoPrefix | CommandFlags.Session)]
		public static void GetRandomUnscannedDocument()
		{
			//While Get random dwg from database that not scanned
			var doc = DbMongo.Instance.GetNewRandomUnscannedDocument();

			if (doc != null)
			{
				// You will only see it once
				DbMongo.Instance.SetDocumentScanned(doc.FileId);

				currentDocument = new CrawlAcDbDocument(doc);
				currentDocument.DoOpen();
			}
		}

		/// <summary>
		/// Writes selection as a fragment to database
		/// </summary>
		[CommandMethod("fra", CommandFlags.NoCheck | CommandFlags.NoPrefix | CommandFlags.Redraw)]
		public static void WriteFragment()
		{
			List<McObjectId> currentSelection = new List<McObjectId>(McObjectManager.SelectionSet.CurrentSelection);
			if (currentSelection.Count == 0)
				currentSelection = McObjectManager.SelectObjects("Select fragment").ToList();

			// каждый набор объектов со своим Guid группы
			var groupId = Guid.NewGuid().ToString();
			CrawlAcDbDocument.DumpFragment(currentSelection, currentDocument.FileId, groupId);
		}

		/// <summary>
		/// Rescans saved fragments to update objects' fileds in database
		/// </summary>
		[CommandMethod("rescan", CommandFlags.NoCheck | CommandFlags.NoPrefix | CommandFlags.Session)]
		public static void Rescan()
		{
			foreach (CrawlDocument doc in DbMongo.Instance.GetAllScannedDocuments())
			{
				currentDocument = new CrawlAcDbDocument(doc);
				currentDocument.DoOpen();

				foreach (long handle in DbMongo.Instance.GetHandlesFromDoc(doc.FileId))
				{
					McObjectId id = McObjectId.FromHandle(handle);
					if (id.IsNull)
						continue;

					var obj = id.GetObject();
					if (obj != null)
					{
						var cent = Converters.From(id);

						DbMongo.Instance.UpdateObject(handle, Converters.Serialize(cent));
					}
				}

				currentDocument.Document.Close();
			}
		}
	}
}
